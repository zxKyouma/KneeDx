"""DisMix arm A -- ordinal-adjacent, per-finding query-vector mixup.

scratch/dismix/PREREG.md. Port of the "Ordinal Mix" policy of DisMix
(Pitawela, Carneiro & Chen, arXiv:2608.04652) WITHOUT the dual-codebook
VQ-VAE: in our model the 12 pre-head study-query vectors are already
per-finding representations (query k feeds head row k only), so query row k
plays the role of the paper's ordinal latent z^o for finding k and the other
11 rows + everything upstream play the non-ordinal latent, which DisMix holds
FIXED from one source while only the ordinal code is interpolated.

Rule, per training study and per finding k, independently with prob p:
  partner = the most recent banked study whose grade on finding k is
            ADJACENT (g-1 or g+1 on the 4-step ladder; never same grade,
            never silent, never a non-ladder target), age <= max_age steps
  lam ~ Beta(alpha, alpha)
  z_mix[k]  = lam * z[k]  + (1 - lam) * z_bank[k]        (bank is DETACHED)
  y_mix[k]  = lam * y[k]  + (1 - lam) * y_bank[k]        (ladder soft targets)
  w_mix[k]  = lam * w[k]  + (1 - lam) * w_bank[k]
  rows that do not mix keep their own tensors EXACTLY (torch.where).
Only the primary soft-BCE sees the mixed logits/targets; the trainer restores
the unmixed tensors for every later consumer (mmix precedent).

How this differs from the KILLED manifold mixup (regbundle_mm_f0, --mmix-p):
mmix mixed the WHOLE 12-row tensor against ONE arbitrary partner (the previous
study) with ONE shared lam, so most mixes were grade-0 vs grade-0 or spanned
non-adjacent grades and moved all findings together. Here each finding mixes
independently, only across one ladder step, with the other 11 rows untouched.

State lives on this object (train-only, parameter-free, never saved in the
checkpoint). With --dismix-p 0 the trainer never constructs it and the
training path is byte-identical to the committed one.
"""
import random

import torch

from model import oa_gate_factorize, osseous_factorize


def head_from_z(model, z):
    """Primary 12-finding logits from (B, n_out, D) query vectors -- the same
    three lines KneeNet.forward applies to the unmixed study_emb."""
    logits = (model.drop(z) * model.head).sum(-1) + model.head_b
    if model.osseous_factor:
        logits = osseous_factorize(logits)
    if model.oa_gate:
        logits = oa_gate_factorize(logits)
    return logits


class DisMix:
    def __init__(self, ladder, n_out, p=0.5, alpha=1.0, max_age=256, seed=0,
                 tol=1e-4):
        self.ladder = torch.tensor([float(v) for v in ladder])   # ascending
        self.n_out = int(n_out)
        self.p = float(p)
        self.alpha = float(alpha)
        self.max_age = int(max_age)
        self.tol = float(tol)
        self.rng = random.Random(seed)
        self.bank = {}        # (finding, grade) -> dict(z, y, w, sh, step)
        self.step = 0
        # counters for the per-epoch print
        self.n_steps = 0
        self.n_steps_mixed = 0
        self.n_cells = [0] * self.n_out
        self.n_eligible = [0] * self.n_out
        self.n_no_partner = [0] * self.n_out

    # ------------------------------------------------------------------
    def grade_index(self, y):
        """(B, n_out) soft targets -> (B, n_out) long ladder index, -1 where
        the target is not (within tol) a ladder value: silent 0.10, cotrain
        zeros, teacher/soft targets never mix."""
        lad = self.ladder.to(y.device)
        d = (y.detach().float().unsqueeze(-1) - lad).abs()
        m, idx = d.min(-1)
        return torch.where(m < self.tol, idx, torch.full_like(idx, -1))

    def _partner(self, k, g):
        c = [(k, g - 1), (k, g + 1)]
        c = [t for t in c if t in self.bank
             and self.step - self.bank[t]["step"] <= self.max_age]
        if not c:
            return None
        return self.bank[self.rng.choice(c)]

    # ------------------------------------------------------------------
    def apply(self, model, z, y, w_cell, sh, ok):
        """z (B, n_out, D) UNMIXED query vectors from the forward just done;
        y (B, n_out) soft targets; w_cell (B, n_out) cell weights; sh
        (B, n_out) sharpen or None; ok (B, n_out) bool = supervised & finite.
        Returns (logits_or_None, y, w, sh, lam_cell, hit)."""
        B, n, _ = z.shape
        gi = self.grade_index(y)
        ok = ok & (gi >= 0)
        lam_cell = torch.ones(B, n, device=z.device, dtype=torch.float32)
        zb = torch.zeros_like(z)
        yb = torch.zeros_like(y)
        wb = torch.zeros_like(w_cell)
        shb = torch.zeros_like(sh) if sh is not None else None
        ok_c, gi_c = ok.cpu(), gi.cpu()
        hit = False
        self.n_steps += 1
        for b in range(B):
            for k in range(n):
                if not bool(ok_c[b, k]):
                    continue
                self.n_eligible[k] += 1
                if self.rng.random() >= self.p:
                    continue
                e = self._partner(k, int(gi_c[b, k]))
                if e is None:
                    self.n_no_partner[k] += 1
                    continue
                lam = self.rng.betavariate(self.alpha, self.alpha)
                lam_cell[b, k] = lam
                zb[b, k] = e["z"].to(z.device, z.dtype)
                yb[b, k] = e["y"]
                wb[b, k] = e["w"]
                if shb is not None and e["sh"] is not None:
                    shb[b, k] = e["sh"]
                self.n_cells[k] += 1
                hit = True
        if not hit:
            return None, y, w_cell, sh, lam_cell, False
        self.n_steps_mixed += 1
        mixed = lam_cell < 1
        lz = lam_cell.to(z.dtype)[..., None]
        zm = torch.where(mixed[..., None], lz * z + (1.0 - lz) * zb, z)
        logits = head_from_z(model, zm)
        ly = lam_cell.to(y.dtype)
        y_m = torch.where(mixed, ly * y + (1.0 - ly) * yb, y)
        lw = lam_cell.to(w_cell.dtype)
        w_m = torch.where(mixed, lw * w_cell + (1.0 - lw) * wb, w_cell)
        sh_m = sh
        if sh is not None:
            ls = lam_cell.to(sh.dtype)
            sh_m = torch.where(mixed, ls * sh + (1.0 - ls) * shb, sh)
        return logits, y_m, w_m, sh_m, lam_cell, True

    def update(self, z, y_raw, w_raw, sh_raw):
        """Bank the current study's UNMIXED rows (detached) under (finding,
        grade) for every cell with a supervised on-ladder target. Called
        every step, mixed or not, after apply()."""
        gi = self.grade_index(y_raw).cpu()
        w_c = w_raw.detach().float().cpu()
        zd = z.detach()
        for b in range(z.shape[0]):
            for k in range(self.n_out):
                g = int(gi[b, k])
                if g < 0 or float(w_c[b, k]) <= 0:
                    continue
                self.bank[(k, g)] = {
                    "z": zd[b, k].clone(),
                    "y": float(y_raw[b, k]),
                    "w": float(w_raw[b, k]),
                    "sh": (float(sh_raw[b, k]) if sh_raw is not None else None),
                    "step": self.step,
                }
        self.step += 1

    def summary(self, names):
        tot = sum(self.n_cells)
        per = " ".join(f"{names[k][:4]}={self.n_cells[k]}/{self.n_eligible[k]}"
                       for k in range(self.n_out))
        return (f"[dismix] steps={self.n_steps} mixed_steps={self.n_steps_mixed} "
                f"mixed_cells={tot} no_partner={sum(self.n_no_partner)} "
                f"bank={len(self.bank)} | {per}")
