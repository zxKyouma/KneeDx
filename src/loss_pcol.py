"""Prototype-based contrastive ordinal loss (PCOL / SCOLw) as an AUXILIARY
term on top of the production soft-target BCE.

Source: "A Hybrid Contrastive Ordinal Regression Method for Advancing Disease
Severity Assessment in Imbalanced Medical Datasets", MICCAI 2025
(scratch/litscan/A_conferences.md #3). The paper's two ingredients:
  (i)  PCOL: pull each sample's embedding toward a LEARNED per-grade prototype
       with a distance-aware ordinal penalty, so grade geometry in embedding
       space respects the ordinal order;
  (ii) SCOLw: inverse-frequency class weighting so rare high grades are not
       swamped by the common low grades.
The exact formulas are not fully retrievable from the accessible text
(flagged in the scan); the construction below is stated explicitly so it can be
audited and is the object the prereg funds.

Embedding hooked: the 12 scored query outputs of the study transformer AFTER
the routing LayerNorm and BEFORE dropout and the per-query linear head, i.e.
`study_emb[:, :12]` in KneeNet.forward (the tensor the head dot-products
against). Shape (B, 12, D). One prototype bank per FINDING: P_k in R^{4 x D}
for grades g0..g3.

Grade mapping from soft targets: a cell is "graded" when its target equals
one of the four ladder values prob_table[0..3] (|y - ladder| < 1e-3), its BCE
weight is > 0, and it does NOT equal the silent value prob_table[-1]. Silent
cells, cotrain-pool zero-weight cells, mixed/teacher targets and mirrored
masks are all excluded from the prototype loss. The ladder is READ from the
run's prob_table, never hardcoded.

Per graded cell (k, g), with cosine similarities s_j = <z_k, P_k[j]> / tau:
    q_j  = softmax_j( -|j - g| / sigma )            ordinal soft target over prototypes
    L    = w_{k,g} * sum_j -q_j log softmax_j(s)_j   distance-aware prototype CE
    w_{k,g} = (n_k / n_{k,g})^0.5, normalised to mean 1 over the four grades
             of finding k (inverse-frequency rare-grade weighting; counts from
             the TRAIN split's graded cells; grades absent from train get the
             max weight of that finding).
Loss = mean over graded cells in the batch (0 when none). sigma=1 gives
q = [.58,.21,.08,.03]-ish for g0 and puts mass on adjacent grades only; the
argmax grade (the one the task statement names) always carries the largest
share. tau = 0.1 as in supervised-contrastive practice.

The prototypes are trained by the SAME optimizer (own param group, no weight
decay, lr = head lr) and live OUTSIDE the model: checkpoints stay byte-
compatible with every consumer. --pcol-w 0 (default) constructs nothing and
executes no extra tensor op.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCOL(nn.Module):
    def __init__(self, n_findings, dim, prob_table, tau=0.1, sigma=1.0,
                 grade_counts=None):
        super().__init__()
        self.n = int(n_findings)
        self.dim = int(dim)
        self.tau = float(tau)
        self.sigma = float(sigma)
        ladder = [float(prob_table[g]) for g in range(4)]
        self.register_buffer("ladder", torch.tensor(ladder))
        sil = prob_table.get(-1, None)
        self.silent = None if sil is None else float(sil)
        if self.silent is not None and min(abs(self.silent - l) for l in ladder) < 2e-3:
            raise SystemExit(f"[pcol] silent value {self.silent} collides with "
                             f"ladder {ladder}; cannot separate silent cells")
        self.proto = nn.Parameter(torch.randn(self.n, 4, self.dim) * 0.02)
        d = torch.arange(4).float()
        # (4 target grades, 4 prototypes): ordinal soft target rows
        self.register_buffer("q", F.softmax(-(d[:, None] - d[None, :]).abs()
                                            / self.sigma, dim=1))
        w = torch.ones(self.n, 4)
        if grade_counts is not None:
            c = np.asarray(grade_counts, dtype=np.float64)   # (n, 4)
            for k in range(self.n):
                nk = c[k].sum()
                if nk <= 0:
                    continue
                wk = np.where(c[k] > 0, np.sqrt(nk / np.maximum(c[k], 1)), np.nan)
                wk = np.where(np.isnan(wk), np.nanmax(wk), wk)
                w[k] = torch.tensor(wk / wk.mean(), dtype=torch.float32)
        self.register_buffer("w", w)

    def grades(self, y, w_cell):
        """(B, n) long grade in 0..3, or -1 where the cell is not graded."""
        yf = y.float()
        dist = (yf.unsqueeze(-1) - self.ladder).abs()          # (B, n, 4)
        mind, g = dist.min(-1)
        ok = (mind < 1e-3) & (w_cell.float() > 0)
        if self.silent is not None:
            ok &= (yf - self.silent).abs() >= 1e-3
        return torch.where(ok, g, torch.full_like(g, -1))

    def forward(self, z, y, w_cell):
        """z (B, n, D) pre-head query embeddings; y (B, n) soft targets;
        w_cell (B, n) BCE cell weights. Returns (loss, n_graded_cells)."""
        g = self.grades(y, w_cell)
        m = g >= 0
        n_ok = int(m.sum().item())
        if n_ok == 0:
            return z.float().sum() * 0.0, 0
        bi, ki = m.nonzero(as_tuple=True)
        zf = F.normalize(z.float()[bi, ki], dim=-1)             # (M, D)
        pf = F.normalize(self.proto[ki].float(), dim=-1)         # (M, 4, D)
        s = torch.einsum("md,mjd->mj", zf, pf) / self.tau        # (M, 4)
        gg = g[bi, ki]
        logp = F.log_softmax(s, dim=1)
        ce = -(self.q[gg] * logp).sum(1)                         # (M,)
        wt = self.w[ki, gg]
        return (wt * ce).sum() / wt.sum(), n_ok


def grade_counts_from_labels(labels, uids, prob_table, n_findings):
    """(n_findings, 4) counts of graded cells over `uids` (train split only)."""
    ladder = np.array([float(prob_table[g]) for g in range(4)])
    sil = prob_table.get(-1, None)
    c = np.zeros((n_findings, 4), dtype=np.int64)
    for u in uids:
        y = labels.get(u)
        if y is None:
            continue
        y = np.asarray(y, dtype=np.float64)[:n_findings]
        d = np.abs(y[:, None] - ladder[None, :])
        g = d.argmin(1)
        ok = d.min(1) < 1e-3
        if sil is not None:
            ok &= np.abs(y - float(sil)) >= 1e-3
        for k in np.nonzero(ok)[0]:
            c[k, g[k]] += 1
    return c
