"""SLICE-WITNESS (slicewit) training-only helpers — scratch/slicewit/PREREG.md.

The series-level WITNESS-EV term (src/train.py ``witness_loss``) supervises the
EvidenceMIL pool ACROSS series. slicewit supervises the pool WITHIN a series:
for every gated (study, series, finding) cell it adds

    c * KL( v_T || v_S ),   v = softmax over the SHARED native slices of
                                (per-slice evidence / tau)

where the per-slice evidence is the raw EvidenceMIL readout
``e = ev_head(tokens)`` (``KneeNet._evidence_forward``: ``(B, S, Z, F)``, the
tensor that the focal LSE pool (after a 3-tap smoothing) and the diffuse
mean/q.8/max mixture pool into the per-series evidence ``out[1]``), and v_T
comes from the honest-OOF EV fold member that HELD THE STUDY OUT.

ALIGNMENT (the load-bearing decision). ``src/data.py::_load_series`` chooses the
24 slices of a series as

    z >  24, train : round(clip(linspace(0, z-1, 24) + U(-.5, .5), 0, z-1))
    z >  24, eval  : round(linspace(0, z-1, 24))
    z <= 24        : clip(arange(24) - (24 - z)//2, 0, z-1)   (edge-replicated)

then REVERSES the stack for a right-knee sagittal series (and again under
flip-swap, which slicewit refuses). The train-time grid is therefore JITTERED
against the teacher's eval grid whenever z > 24, and slot j is not the same
native slice in teacher and student. The witness is keyed by the NATIVE slice
index (the row of the cached volume, identical for teacher and student because
both read the same .npy) and BOTH softmaxes are taken over the native slices
present in BOTH views, a duplicated native slice (z < 24) contributing the mean
of its slots' evidence on each side. For z <= 24 the two views are identical
slot for slot, so the alignment is exact on every slice.

The student's native indices are recovered WITHOUT touching src/data.py and
WITHOUT consuming the RNG stream: ``SliceIndexRecorder._load_series`` snapshots
``random.getstate()`` immediately before calling the real ``_load_series`` (in
which the grid jitter ``random.uniform(-0.5, 0.5)`` is the FIRST random draw)
and replays that one draw on a PRIVATE ``random.Random`` instance.
"""
import math
import os
import random

import numpy as np
import torch


def native_slice_index(z, n_slices, off=None):
    """Native (cached-volume) row of each of the n_slices slots, BEFORE any
    stack reversal. Verbatim arithmetic of src/data.py::_load_series (numpy
    ``round`` = half-to-even, as there). ``off`` = the train-time jitter draw,
    None in eval mode."""
    z, n = int(z), int(n_slices)
    if z > n:
        if off is None:
            idx = np.linspace(0, z - 1, n)
        else:
            idx = np.clip(np.linspace(0, z - 1, n) + off, 0, z - 1)
        idx = idx.round().astype(int)
    else:
        idx = np.clip(np.arange(n) - (n - z) // 2, 0, z - 1)
    return np.asarray(idx, np.int64)


class SliceIndexRecorder:
    """Mixin (put it FIRST in the bases) that records, per ``_load_series``
    call made while ``self._sw_rec`` is a list, ``(series_uid, native index per
    slot AFTER reversal)``. It adds one header-only ``np.load(mmap)`` per series
    and no call to the module-level ``random``/``np.random`` streams, so the
    pixels, labels and RNG stream are byte-identical to the base class (unit
    tested). Refused configurations (checked by the trainer/extractor, and
    asserted here where cheap): stratified_slices, evidence_sampling,
    native_ctx, flip-swap."""

    _sw_rec = None

    def _load_series(self, study, series_uid, plane=None, flip=False,
                     want_mask=False, mask_path=None):
        st = random.getstate()
        v = super()._load_series(study, series_uid, plane, flip=flip,
                                 want_mask=want_mask, mask_path=mask_path)
        if self._sw_rec is not None:
            self._sw_rec.append(
                (series_uid, self._sw_native(study, series_uid, plane, flip, st)))
        return v

    def _sw_native(self, study, series_uid, plane, flip, st):
        n = int(self.n_slices)
        if getattr(self, "stratified_slices", False) or \
                getattr(self, "evidence_sampling", False) or \
                getattr(self, "native_ctx", False):
            raise RuntimeError("slicewit: stratified_slices / evidence_sampling "
                               "/ native_ctx change the slice grid and are "
                               "refused")
        path = os.path.join(self.vol_root, study, series_uid + ".npy")
        try:
            z = int(np.load(path, mmap_mode="r").shape[0])
        except Exception:
            return np.full(n, -1, np.int64)       # zeros volume: no witness
        off = None
        if z > n and self.train:
            r = random.Random()
            r.setstate(st)
            off = r.uniform(-0.5, 0.5)            # == the draw data.py made
        nat = native_slice_index(z, n, off)
        rev = False
        if plane == "Sagittal":
            rev = (self.side.get(study) == "R") != bool(flip)
        return nat[::-1].copy() if rev else nat


def load_slicewit_teacher(path, n_findings):
    """Combined npz -> ({study: {series_uid: (nat (Z,), e (Z, F))}}, sfold)."""
    z = np.load(path, allow_pickle=True)
    e = z["e"].astype(np.float32)
    nat = z["nat"].astype(np.int64)
    ser = [str(s) for s in z["ser"]]
    offs = z["offs"].astype(np.int64)
    if e.ndim != 3 or e.shape[2] != n_findings or nat.shape != e.shape[:2]:
        raise SystemExit(f"slicewit teacher {path}: bad shapes e{e.shape} "
                         f"nat{nat.shape}")
    teacher, sfold = {}, {}
    for i, u in enumerate(z["uids"]):
        u = str(u)
        teacher[u] = {ser[j]: (nat[j], e[j])
                      for j in range(int(offs[i]), int(offs[i + 1]))}
        sfold[u] = int(z["sfold"][i])
    return teacher, sfold


def _avg_matrix(nat, shared):
    a = (nat[None, :] == shared[:, None]).astype(np.float64)
    return a / a.sum(1, keepdims=True)


def slicewit_loss(e, mask, studies, wit_ser, wit_sli, teacher, tau, col_idx,
                  target):
    """Gate-weighted mean slice-witness KL.

    e        (B, S, Z, F) student per-slice evidence (ev_head output, float).
    mask     (B, S)       1 = a real series in that slot.
    studies  list[str];  wit_ser  list[str] "|"-joined series UID per slot.
    wit_sli  (B, S, Z)    native slice index per slot (-1 = none).
    teacher  {study: {series_uid: (nat (Z,), e_T (Z, F))}}.
    col_idx  LongTensor of the scored columns (Synovitis NOT in it: it is never
             indexed, so its gradient is structurally zero).
    target   (B, F) the blended study target; DETACHED here.

    Gate (entropy_conf, per (study, series, finding)):
        c = (1 - H(v_T) / log K) * t(i, k),  K = shared native slices >= 2.
    Returns (L, n_cells) with L = sum c*KL / sum c, or (None, 0).
    """
    cols = col_idx.detach().cpu().numpy()
    total = e.new_zeros((), dtype=torch.float32)
    wsum = e.new_zeros((), dtype=torch.float32)
    n_cells = 0
    sli = wit_sli.detach().cpu().numpy() if torch.is_tensor(wit_sli) \
        else np.asarray(wit_sli)
    tgt = target.detach().float()
    for bi, su in enumerate(studies):
        rec = teacher.get(su)
        if rec is None:
            continue
        for j, s in enumerate(wit_ser[bi].split("|")):
            if not s or float(mask[bi, j]) <= 0:
                continue
            tr = rec.get(s)
            if tr is None:
                continue
            nat_t, e_t = tr
            nat_s = sli[bi, j]
            shared = np.intersect1d(nat_s[nat_s >= 0], nat_t[nat_t >= 0])
            k = int(shared.size)
            if k < 2:
                continue
            a_s = torch.as_tensor(_avg_matrix(nat_s, shared), dtype=torch.float32,
                                  device=e.device)                     # (K, Z)
            te = torch.as_tensor(_avg_matrix(nat_t, shared) @
                                 e_t[:, cols].astype(np.float64),
                                 dtype=torch.float32, device=e.device)  # (K, C)
            se = a_s @ e[bi, j][:, col_idx].float()                    # (K, C)
            lt = torch.log_softmax(te / tau, dim=0)
            ls = torch.log_softmax(se / tau, dim=0)
            wt = lt.exp()
            kl = (wt * (lt - ls)).sum(0)                                # (C,)
            h = -(wt * lt).sum(0)
            c = (1.0 - h / math.log(float(k))).clamp(0.0, 1.0)
            c = c * tgt[bi, col_idx].clamp(0.0, 1.0)
            total = total + (c * kl).sum()
            wsum = wsum + c.sum()
            n_cells += len(cols)
    if n_cells == 0 or float(wsum) <= 0.0:
        return None, 0
    return total / wsum, n_cells


class EvHeadTap:
    """Forward hook on ``model.ev_head`` capturing the per-slice evidence of
    the ARMED forward only. Output is never modified (the hook returns None),
    so the forward, the returned tensors and the graph are unchanged."""

    def __init__(self, module):
        self.armed = False
        self.out = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.armed:
            self.out = out

    def arm(self):
        self.out = None
        self.armed = True

    def take(self, B, S, Z):
        self.armed = False
        o, self.out = self.out, None
        if o is None:
            return None
        return o.reshape(B, S, Z, o.shape[-1]).float()
