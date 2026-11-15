"""Train the study-level knee model on report-derived soft labels.

Validation is deliberately two-track:

- **Fold AUC** against binarised report labels. Plenty of studies, tight error
  bars, but it only measures agreement with the report - which is *not* what
  the competition scores.
- **Gold AUC** against the 58 expert-labelled studies. This is the real target
  and the only unbiased estimate we have, but n=58 makes per-finding AUC very
  noisy (roughly +/-0.06). Treat it as a directional check, never tune on it
  aggressively.

The 58 gold studies are held out of training entirely so that second number
stays honest.
"""

import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (FINDINGS, KneeStudyDataset, fit_prob_table, gold_calib_split, load_zspacing, load_aspect,
                  load_geometry, load_gold, load_laterality, load_report_labels,
                  load_sequence, SEQ_CLASSES, load_aux_labels, load_agreement,
                  load_series_index, near_duplicate_groups, fit_silent_table,
                  load_label_repair_sidecar, training_label_repairs,
                  load_evidence_sampling_map)
from model import KneeNet, soft_bce
from slicewit import (SliceIndexRecorder, EvHeadTap,  # noqa: E402  scratch/slicewit
                      load_slicewit_teacher, slicewit_loss)


def auc_score(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    i = 0
    ss = s[order]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def freeze_bn_modules(model):
    """--freeze-bn: BatchNorm* modules of the backbone use running stats
    (eval mode) while everything else stays in train mode. Returns the count.
    Must be re-applied after every model.train() call, which resets them."""
    n = 0
    for m in model.backbone.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval(); n += 1
    return n


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    P, Y, U = [], [], []
    with torch.no_grad():
        for b in loader:
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(b["vol"].to(device, non_blocking=True),
                            b["stype"].to(device),
                            b["mask"].to(device),
                            b["seq"].to(device))
            P.append(torch.sigmoid(out[0].float()).cpu().numpy())
            if "y" in b:
                Y.append(b["y"].numpy())
            U += list(b["study"])
    P = np.concatenate(P)
    Y = np.concatenate(Y) if Y else None
    # Neither AUC function rejects NaN: auc_score's argsort ranks NaN highest,
    # and soft_auc's comparisons are all-False so those rows sit in the
    # denominator winning nothing. Injecting 12% NaN into one column returns a
    # clean-looking 0.49 with no warning, so an fp16 blow-up would show up as a
    # mediocre epoch rather than as a failure. Say so instead.
    n_bad = int((~np.isfinite(P)).sum())
    if n_bad:
        print(f"   WARNING {n_bad}/{P.size} non-finite predictions "
              f"({n_bad / P.size:.2%}) - AUCs below are not meaningful",
              flush=True)
    return P, Y, U


def bce_and_floors(P, Y):
    """Validation BCE next to the two reference points that make it readable.

    Training loss is measured with dropout and augmentation on, so it is not
    comparable to a clean floor. These three numbers are:
      loss  - what the model actually achieves on held-out data
      base  - predicting each finding's base rate, i.e. zero image information
      floor - the entropy of the soft targets, which no model can beat
    """
    eps = 1e-9
    bce = lambda p, y: float(-(y * np.log(np.clip(p, eps, 1)) +
                               (1 - y) * np.log(np.clip(1 - p, eps, 1))).mean())
    prior = Y.mean(0, keepdims=True)
    return (bce(P, Y), bce(np.repeat(prior, len(Y), 0), Y),
            bce(Y, Y))


def soft_auc(p, s):
    """AUC against soft labels: P(score_pos > score_neg) weighted by p_i(1-p_j).

    Thresholding the soft targets at 0.5 puts grade 1 (0.358) in the same bucket
    as grade 0 (0.056), and AUC cannot see how negatives rank among themselves -
    so the selector was deaf to the grade-1 vs grade-0 separation that is the
    main advantage over a binary extractor.
    """
    p = np.asarray(p, float)
    s = np.asarray(s, float)
    q = 1.0 - p
    denom = p.sum() * q.sum()
    if denom <= 0:
        return float("nan")
    gt = (s[:, None] > s[None, :]).astype(float)
    eq = (s[:, None] == s[None, :]).astype(float)
    num = (p[:, None] * q[None, :] * (gt + 0.5 * eq)).sum()
    return float(num / denom)


def soft_auc_ceiling(y):
    """Best soft AUC any ranker can reach on these targets: rank by the target.

    Soft AUC cannot reach 1.0. Two studies both at p=0.35 contribute weight
    p(1-p) in BOTH directions, so even a perfect ranker scores 0.5 on that
    pair. The ceiling therefore depends on how peaked a finding's targets are.
    """
    return soft_auc(y, y)


def mean_soft_auc(P, Y, normalise=False, cols=None):
    """Mean soft AUC over findings, or over `cols` if given.

    `cols` exists for specialist arms. A trunk trained on one finding still
    emits twelve columns, eleven of which never received a gradient, so an
    all-column selection metric would be 11/12 noise and pick its "best"
    checkpoint essentially at random. Default None keeps every previous run
    bit-identical.

    normalise=True rescales each finding to the fraction of its ACHIEVABLE
    range, (auc-0.5)/(ceiling-0.5), before averaging. This matters for
    checkpoint selection: measured on the real fold-0 validation set the
    ceilings run 0.7465 (Fracture) to 0.8894 (ACL), mean 0.8369, so an
    unweighted mean gives ACL 1.58x the influence of Fracture per unit of true
    skill - biasing selection away from Fracture and Synovitis, the two
    findings with the most headroom. The competition scores twelve equally
    weighted hard AUCs, so equal normalised weight is the closer proxy.
    """
    vals = []
    for k in (range(P.shape[1]) if cols is None else cols):
        a = soft_auc(Y[:, k], P[:, k])
        if normalise and np.isfinite(a):
            c = soft_auc_ceiling(Y[:, k])
            if np.isfinite(c) and c > 0.5 + 1e-9:
                a = (a - 0.5) / (c - 0.5)
        vals.append(a)
    return float(np.nanmean(vals))


def mean_auc(P, Y, thresh=0.5):
    aucs = []
    for k in range(P.shape[1]):
        yk = (Y[:, k] >= thresh).astype(int)
        aucs.append(auc_score(yk, P[:, k]))
    return float(np.nanmean(aucs)), aucs


def teacher_curriculum_weight(schedule, epoch_float, max_weight=0.5,
                              ramp_start=2.0, ramp_end=5.0):
    """Return the live report/teacher target blend for one training step.

    ``epoch_float`` is the fractional epoch at the actual micro-step, rather
    than the integer epoch boundary.  The late arm therefore starts with pure
    report targets, ramps linearly from epoch 2 to 5, and stays at 0.5 after
    epoch 5.  Keeping this as a side-effect-free function makes the schedule
    easy to test without constructing a CUDA model or a DataLoader.
    """
    if schedule in (None, "off", "none"):
        return 0.0
    if schedule == "fixed":
        return float(max_weight)
    if schedule == "late":
        if epoch_float < ramp_start:
            return 0.0
        if epoch_float >= ramp_end:
            return float(max_weight)
        return float(max_weight) * ((epoch_float - ramp_start) /
                                    (ramp_end - ramp_start))
    raise ValueError(f"unknown teacher curriculum {schedule!r}")


def label_repair_bce_elements(logits, target, lo, hi, interval):
    """Elementwise BCE with set-valued targets on selected cells.

    An interval cell has zero loss while the predicted probability lies in
    ``[lo, hi]``. Below or above it, BCE is evaluated against the nearest
    endpoint. Ordinary cells retain BCE against ``target`` exactly.
    """
    base = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits.float())
    nearest = torch.where(p < lo, lo, hi).to(target.dtype)
    endpoint = F.binary_cross_entropy_with_logits(
        logits, nearest, reduction="none")
    # Bernoulli KL(prediction || interval endpoint) up to orientation: BCE
    # minus the endpoint entropy. This retains the BCE gradient while making
    # the set-valued loss continuous and exactly zero at either boundary.
    q = nearest.clamp(1e-7, 1 - 1e-7)
    entropy = -(q * torch.log(q) + (1 - q) * torch.log1p(-q))
    endpoint = (endpoint - entropy).clamp_min(0)
    outside = (p < lo) | (p > hi)
    interval_loss = torch.where(outside, endpoint, torch.zeros_like(endpoint))
    return torch.where(interval, interval_loss, base)


def label_repair_bce(logits, target, lo, hi, interval, weight=None,
                     sharpen=None):
    """Mean label-repair loss, matching ``soft_bce`` off interval cells."""
    if not bool(interval.any()):
        return soft_bce(logits, target, weight=weight, sharpen=sharpen)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if sharpen is not None:
        z = logits.float()
        p = torch.sigmoid(z)
        h = -(p * F.logsigmoid(z) + (1.0 - p) * F.logsigmoid(-z))
        loss = ((1.0 + sharpen) * loss
                - sharpen * (math.log(2.0) - h).to(loss.dtype))
    interval_loss = label_repair_bce_elements(
        logits, target, lo, hi, interval)
    loss = torch.where(interval, interval_loss, loss)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def batch_label_bce(logits, target, batch, weight=None, sharpen=None):
    """Dispatch to the old loss unless a training batch carries repairs."""
    if "lr_interval" not in batch:
        return soft_bce(logits, target, weight=weight, sharpen=sharpen)
    return label_repair_bce(
        logits, target,
        batch["lr_lo"].to(logits.device, non_blocking=True),
        batch["lr_hi"].to(logits.device, non_blocking=True),
        batch["lr_interval"].to(logits.device, non_blocking=True),
        weight=weight, sharpen=sharpen)


def aux_seg_ce_dice(logits, target, n_classes=6, eps=1e-6):
    """--aux-seg-w (scratch/auxdice/PREREG.md): CE + soft Dice, Dice
    averaged only over classes PRESENT in this window's target (a window
    with no lateral-cartilage pixels should not be rewarded for predicting
    none, nor penalised for a class that is genuinely absent here).
    logits: (N, C, H, W) float. target: (N, H, W) long in [0, C)."""
    ce = F.cross_entropy(logits, target)
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    present = onehot.sum(dim=(0, 2, 3)) > 0
    if present.any():
        inter = (probs * onehot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        dice = (2 * inter + eps) / (union + eps)
        dice_loss = 1.0 - dice[present].mean()
    else:
        dice_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    return ce + dice_loss


def bagauc_pair_loss(z, pi):
    """Cross-bag pairwise logistic loss for ONE finding, ONE window.

    z: (n,) logits - LIVE, i.e. every entry must still be attached to the
    graph of the forward pass that produced it (never a detached bank; see
    --rank-w's help string for the killed arm this must not become). pi:
    (n,) calibrated bag prevalence aligned with z - two entries share a bag
    iff their pi is equal, so bag identity never needs to be passed
    separately, and a study whose grade could not be resolved should simply
    be excluded by the caller before this function ever sees it.

    L = mean over pairs with pi_row > pi_col of
        (pi_row - pi_col) * softplus(-(z_row - z_col))

    Properties (verified in scratch/bagauc/test_loss.py):
      (a) two same-bag entries -> PI=0 between them -> zero contribution and
          zero relative gradient between their z's.
      (b) an inverted cross-bag pair (higher-pi entry has the LOWER z) gets
          a gradient that pushes z_row - z_col UP - genuine reordering,
          because both z_row and z_col are live in the same backward call.
      (c) if every entry shares one bag, PI is zero everywhere and the loss
          is exactly 0 (returns None so the caller adds nothing).

    Returns (loss or None, n_pairs). n_pairs counts only PI>1e-9 entries, so
    it is the number of pairs that actually contribute gradient this call -
    the "pair-count statistics" the PREREG requires logged per epoch.
    """
    Z = z.unsqueeze(1) - z.unsqueeze(0)     # (n,n) z_row - z_col
    PI = pi.unsqueeze(1) - pi.unsqueeze(0)  # (n,n) pi_row - pi_col, no grad
    pos = PI > 1e-9
    n_pairs = int(pos.sum().item())
    if n_pairs == 0:
        return None, 0
    loss = (F.softplus(-Z) * PI.clamp(min=0)).sum() / n_pairs
    return loss, n_pairs


BILAT_STRATA = ("STRONG", "MILD")


def bilat_pair_penalty(z_l, z_r, cols, dirs, strata):
    """Softplus (soft-margin, margin 0) L-vs-R ranking penalty for ONE
    bilateral pair (--bilat-rank, scratch/oaibilat/PREREG.md).

    z_l, z_r: (12,) LIVE challenge logits of the left / right knee - both
    must come from forwards in the SAME graph, so one backward reorders the
    two knees jointly. cols: (K,) long FINDINGS indices of this pair's
    DISCORDANT OA compartments (K >= 1; ties never reach this function -
    excluded at build time). dirs: (K,) float, +1 if the LEFT knee is the
    rubric-worse knee on that compartment, -1 if the RIGHT one (a knee can
    be worse medially and better laterally, so direction is per-column).
    strata: (K,) long, index into BILAT_STRATA (0=STRONG >=2 ordinal steps,
    1=MILD one step).

    term_k = softplus(z_better - z_worse): the zero-margin logistic ranking
    loss - softplus supplies the soft margin; the gradient pushes the worse
    knee's logit UP and the better knee's DOWN until worse > better, and
    decays smoothly once the ordering is satisfied.

    Returns (pen, mass, counts): pen = sum(term)/K, the scalar that trains
    (weighted by --bilat-rank upstream); mass[stratum] = detached
    sum(term)/K restricted to the stratum - i.e. the stratum's ACTUAL
    contribution to pen, accumulated per epoch for the frozen
    ">=60% of ranking-loss mass in STRONG" falsification read;
    counts[stratum] = number of comp-terms.
    """
    diff = (z_l[cols] - z_r[cols]) * dirs        # z_worse - z_better
    terms = F.softplus(-diff)
    K = terms.numel()
    pen = terms.sum() / K
    mass, counts = {}, {}
    for si, name in enumerate(BILAT_STRATA):
        m = strata == si
        n = int(m.sum())
        if n:
            mass[name] = float(terms[m].sum().detach()) / K
            counts[name] = n
    return pen, mass, counts


def witness_loss(aux, mask, studies, wit_ser, witness, tau, gate, cols,
                 ncol, target=None):
    """WITNESS-EV training term (scratch/witness/PREREG.md).

    aux       (B, S, F) the model's per-series evidence, out[1].
    mask      (B, S)    1 = a real series in that slot.
    studies   list[str] the batch's study UIDs.
    wit_ser   list[str] per sample, "|"-joined series UIDs, one per SLOT,
                        "" for a padded slot (WitnessStudyDataset).
    witness   {study: ({series_uid: row}, ev (M, F) float32 numpy)}.
    tau       one temperature for teacher AND student.
    gate      "none"          -> c = 1
              "entropy"       -> c = 1 - H(w_T)/log(k)   (concentration)
              "entropy_conf"  -> c = (1 - H(w_T)/log(k)) * t, t = the BLENDED
                                 study target for that cell (collaborator
                                 review 2026-09-22: a low-entropy witness on a
                                 negative study is noise, so the term must
                                 weight positive-evidence cells).
    cols      (F,) column weights; Synovitis is 0 (the placebo column).
    ncol      number of non-zero columns, the per-study cell count.
    target    (B, F) the blended study target y (.5 ladder + .5 teacher under
              this arm's recipe). Required by gate "entropy_conf" only.

    Returns (L, n_cells): L is the GATE-WEIGHTED mean over eligible cells,
    sum_cells c*KL / sum_cells c, or None when no study in the batch
    contributed any gate mass, and n_cells the number of eligible cells. The
    weighted mean (rather than dividing by the cell COUNT) keeps the loss scale
    independent of how much gate mass the teacher happens to carry, so beta
    prices one unit of KL and not the teacher's diffuseness.

    Alignment is by SERIES UID, never by slot: the training loader shuffles
    and subsamples the series every epoch (src/data.py _pick_series), so slot j
    is a different series on different steps. Both softmaxes are taken over the
    SAME set - the slots that are present (mask > 0) AND covered by the witness
    file - so the teacher distribution is renormalised over exactly the series
    the student actually saw. A study with fewer than two such slots
    contributes nothing (the softmax over one series is the constant 1 and the
    KL is identically zero).
    """
    n_cells, total = 0, aux.new_zeros(())
    wsum = aux.new_zeros(())
    for bi, su in enumerate(studies):
        rec = witness.get(su)
        if rec is None:
            continue                     # OAI pool / gold rows: no witness
        idx, ev_all = rec
        slots, rows = [], []
        for j, s in enumerate(wit_ser[bi].split("|")):
            if not s or float(mask[bi, j]) <= 0:
                continue
            r = idx.get(s)
            if r is None:
                continue
            slots.append(j)
            rows.append(r)
        if len(slots) < 2:
            continue
        te = torch.as_tensor(ev_all[rows], dtype=torch.float32,
                             device=aux.device)              # (k, F)
        se = aux[bi, slots].float()                          # (k, F)
        lt = torch.log_softmax(te / tau, dim=0)
        ls = torch.log_softmax(se / tau, dim=0)
        wt = lt.exp()
        kl = (wt * (lt - ls)).sum(0)                         # (F,)
        if gate in ("entropy", "entropy_conf"):
            h = -(wt * lt).sum(0)
            c = (1.0 - h / math.log(float(len(slots)))).clamp(0.0, 1.0)
            if gate == "entropy_conf":
                if target is None:
                    raise RuntimeError("witness gate entropy_conf needs the "
                                       "blended study target")
                c = c * target[bi].detach().to(c.dtype).clamp(0.0, 1.0)
        elif gate == "none":
            c = torch.ones_like(kl)
        else:
            raise RuntimeError(f"unknown witness gate {gate!r}")
        cw = c * cols.to(kl.dtype)
        total = total + (cw * kl).sum()
        wsum = wsum + cw.sum().detach()
        n_cells += ncol
    if n_cells == 0 or float(wsum) <= 0.0:
        return None, 0
    return total / wsum, n_cells


class WitnessStudyDataset(KneeStudyDataset):
    """KneeStudyDataset that also reports WHICH series landed in which slot.

    scratch/witness/PREREG.md. The training loader SHUFFLES and SUBSAMPLES the
    series (src/data.py ``_pick_series``: ``random.shuffle(order)`` and
    ``random.shuffle(by_type[k])`` when ``train``), so slot j is a different
    series on different epochs and a slot-keyed witness target would be
    misaligned on nearly every step. The witness is therefore keyed by SERIES
    UID and aligned at train time; this subclass is the only plumbing needed to
    do that, and it lives HERE rather than in src/data.py so the dataset module
    is untouched and no other arm's batch gains a key.

    ``wit_ser`` is a single "|"-joined string of exactly ``n_series`` slot
    entries (empty string for a padded slot), so ``default_collate`` handles it
    exactly like ``study`` at any batch size. TRAIN-ONLY: the class is used only
    when ``--witness-targets`` is set, and it adds nothing to the model, the
    checkpoint, or the eval path.
    """

    def __getitem__(self, i):
        self._wit_pick = None
        out = super().__getitem__(i)
        pick = self._wit_pick or []
        self._wit_pick = None
        if len(pick) > self.n_series:
            raise RuntimeError(f"witness: {len(pick)} picked series exceeds "
                               f"n_series={self.n_series}")
        out["wit_ser"] = "|".join(list(pick)
                                  + [""] * (self.n_series - len(pick)))
        return out

    def _pick_series(self, study):
        picked = super()._pick_series(study)
        # cutmix would call this a second time for the PARTNER study; --cutmix
        # is refused alongside --witness-targets, and the guard keeps the FIRST
        # (= this item's own) pick regardless.
        if getattr(self, "_wit_pick", None) is None:
            self._wit_pick = [s[0] for s in picked]
        return picked


class SliceWitStudyDataset(SliceIndexRecorder, WitnessStudyDataset):
    """WitnessStudyDataset that ALSO reports the NATIVE slice index of every
    slot (scratch/slicewit/PREREG.md). Adds one key, ``wit_sli`` (n_series,
    n_slices) int64, -1 for a padded slot; pixels, labels and the RNG stream are
    byte-identical to WitnessStudyDataset (src/slicewit.py SliceIndexRecorder,
    unit-tested). TRAIN-ONLY: used only when --slicewit-targets is set."""

    def __getitem__(self, i):
        self._sw_rec = []
        try:
            out = super().__getitem__(i)
            rec = self._sw_rec
        finally:
            self._sw_rec = None
        slots = out["wit_ser"].split("|")
        present = [x for x in slots if x]
        if [r[0] for r in rec] != present:
            raise RuntimeError(f"slicewit: recorded series {[r[0] for r in rec]} "
                               f"!= slot series {present}")
        arr = np.full((self.n_series, self.n_slices), -1, np.int64)
        for j, (_uid, nat) in enumerate(rec):
            arr[j] = nat
        out["wit_sli"] = torch.from_numpy(arr)
        return out


class EMA:
    """Exponential moving average of the weights, evaluated alongside the raw model.

    Averaging the trajectory rather than taking its last point is standard in
    this competition's public kernels (decays of 0.995-0.999) and is absent
    here. It costs one float32 shadow copy and no backward-pass compute.

    The decay is ramped in as (1+n)/(10+n): at step 1 a fixed 0.999 average is
    still 99.9% random initialisation, and it takes thousands of steps to
    forget that. The ramp makes the early EMA usable instead of garbage.

    Consequence worth knowing: the ramp only reaches 0.999 at step 8990, and
    the standard config (bs=1, accum=8, 3489 batches, 12 epochs) takes just
    5232 optimiser steps. So --ema-decay 0.999, 0.9999 and 1.0 give bit-
    identical results; only values below ~0.998 bind at all. The effective
    final decay is ~0.9983, a window of ~580 steps or 1.3 epochs. Use
    effective_decay() rather than trusting the flag.
    """

    def __init__(self, model, decay, ramp=10.0):
        # ramp is the warmup denominator in min(decay, (1+n)/(ramp+n)).
        # The incumbent 10.0 caps effective decay at ~0.9983 (12-ep plain) /
        # 0.999 (10-ep cotrain), which makes every decay above that
        # UNREACHABLE through the flag: the long-window half of the axis has
        # never been testable. A smaller ramp lets the stated decay bind
        # mid-run (n* = (ramp*decay-1)/(1-decay) steps), e.g. 0.9995@2 binds
        # from step ~2000 of a 10830-step cotrain run.
        self.decay = decay
        self.ramp = ramp
        self.n = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (self.ramp + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def effective_decay(self, total_steps):
        """What the decay will actually be at the end of a run of this length."""
        return min(self.decay, (1.0 + total_steps) / (self.ramp + total_steps))

    def state_dict(self, model):
        """A full state_dict: averaged floats, non-float entries passed through."""
        return {k: (self.shadow[k].to(v.dtype) if k in self.shadow
                    else v.detach().clone())
                for k, v in model.state_dict().items()}


_CONDRES = {}


def _condres_step(args, logits, y, studies, loss):
    """CONDRES-R+ (scratch/decouple_20261009/PREREG.md). For each pair T<C and
    each RSNA row whose report target y_T <= thr (report-T-negative: g0,
    silent, low Opus prob), ADD w/12 * rho * zhat_T * sg(zhat_C), where
    zhat = (z - bank mean) / bank sd and rho = Pearson corr(z_T, z_C) over a
    detached FIFO bank of the last N such cells. Gradient flows ONLY into
    z_T (C is stop-grad), it is zero until the bank holds >= 64 cells, and it
    vanishes as the bank correlation -> 0 (self-limiting, no fixed beta).
    OAI/cotrain rows (uid not RSNA) never enter. Returns the new loss."""
    st = _CONDRES
    if not st:
        names = list(FINDINGS)
        pairs = []
        for tok in str(getattr(args, "condres_pairs", "")).split(";"):
            tok = tok.strip()
            if not tok:
                continue
            t, c = [x.strip() for x in tok.split("<")]
            if t not in names or c not in names or t == c:
                raise ValueError(f"--condres-pairs: bad pair {tok!r}")
            pairs.append((names.index(t), names.index(c), tok))
        if not pairs:
            raise ValueError("--condres-w > 0 but --condres-pairs is empty")
        st["pairs"] = pairs
        st["thr"] = float(getattr(args, "condres_thr", 0.15))
        st["n"] = int(getattr(args, "condres_bank", 256))
        st["bank"] = {k: [] for _, _, k in pairs}
        st["step"] = 0
        st["first_rho"] = {}
        print(f"[condres] w={args.condres_w} thr={st['thr']} bank={st['n']} "
              f"pairs={[k for _, _, k in pairs]}", flush=True)
    rs = [i for i, u in enumerate(studies)
          if str(u).startswith("1.2.826.0.1.3680043.8.498.")]
    z = logits.float()
    yt = y.float()
    pen = z.new_zeros(())
    nterm = 0
    for ti, ci, key in st["pairs"]:
        bank = st["bank"][key]
        for r in rs:
            if float(yt[r, ti]) > st["thr"]:
                continue
            zt, zc = z[r, ti], z[r, ci].detach()
            if len(bank) >= 64:
                bt = torch.tensor([a for a, _ in bank], device=z.device)
                bc = torch.tensor([c for _, c in bank], device=z.device)
                mt, mc = bt.mean(), bc.mean()
                sdt, sdc = bt.std().clamp_min(1e-3), bc.std().clamp_min(1e-3)
                rho = (((bt - mt) * (bc - mc)).mean() / (sdt * sdc)).detach()
                if key not in st["first_rho"]:
                    st["first_rho"][key] = float(rho)
                st.setdefault("last_rho", {})[key] = float(rho)
                pen = pen + rho * ((zt - mt) / sdt) * ((zc - mc) / sdc)
                nterm += 1
            bank.append((float(zt.detach()), float(zc)))
            if len(bank) > st["n"]:
                del bank[0]
    st["step"] += 1
    if st["step"] in (1, 100, 1000) or st["step"] % 4000 == 0:
        print("[condres] step %d rho first->now %s" % (st["step"], " ".join(
            f"{k}:{st['first_rho'].get(k, float('nan')):+.3f}->"
            f"{st.get('last_rho', {}).get(k, float('nan')):+.3f}"
            for _, _, k in st["pairs"])), flush=True)
    if nterm == 0:
        return loss
    return loss + (args.condres_w / len(FINDINGS)) * pen.to(loss.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/processed/report_labels.jsonl")
    ap.add_argument("--label-repair-sidecar", default="",
                    help="optional JSONL of audited per-cell training-only "
                         "target, weight, or interval repairs; validated "
                         "against --labels and excluded from validation")
    ap.add_argument("--label-repair-receipt", default="",
                    help="write the verified post-split sidecar application "
                         "count and UID list to this experimental receipt")
    ap.add_argument("--label-repair-expected-cells", type=int, default=0,
                    help="abort if the post-split applied sidecar count differs")
    ap.add_argument("--allow-unpromoted-label-repair", action="store_true",
                    help="allow a candidate sidecar only for its preregistered "
                         "column-only two-seed assay")
    ap.add_argument("--label-repair-promotion", default="",
                    help="detached receipt required for a promoted "
                         "direct_sol_repair_v2 sidecar")
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--series-csv", default="data/raw/train_series.csv")
    ap.add_argument("--vol-root", default="data/volumes/train")
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--backbone-ckpt", default="",
                    help="SSL continued-pretraining backbone state dict "
                         "(scratch/sslpre/); loaded strict over the timm "
                         "backbone after construction")
    ap.add_argument("--resume-init-only", action="store_true",
                    help="noisystud curriculum (scratch/noisystud/PREREG_curric.md): with --resume-model, "
                         "exempt ONLY the data-path keys (labels, vol_root, series_csv, cotrain_pool, "
                         "laterality_map) from the resume contract gate, so a stage-1 model can initialise a "
                         "stage-2 run on a different label/volume set. Model-contract keys stay enforced. "
                         "Default off = the gate is unchanged.")
    ap.add_argument("--resume-model", default="",
                    help="FULL-model resume: load ck['model'] (any checkpoint "
                         "this trainer writes, incl. sub-epoch EMA snapshots) "
                         "strict over the freshly built model, and start "
                         "training from there. Distinct from --backbone-ckpt, "
                         "which loads the backbone only. The checkpoint's "
                         "args are gated against this run's on the contract "
                         "keys, because a resume across a different input "
                         "contract silently trains a mismatched readout.")
    ap.add_argument("--freeze", default="",
                    help="Comma list of parameter partitions to hold FIXED "
                         "(requires_grad=False AND excluded from the "
                         "optimizer, so weight decay cannot move them "
                         "either). Partitions are exactly the checkpoint-"
                         "surgery ones (scratch/gradown/surgery.py): "
                         "backbone | trunkrest (series_enc.* + study_enc.tr.* "
                         "+ study_enc.type_emb.* + aux_head.*) | readout "
                         "(head, head_b, study_enc.queries, study_enc.route.*,"
                         " study_enc.norm.*). --freeze backbone,trunkrest is "
                         "the frztrunk arm: an immutable trunk with a "
                         "co-adapting readout (scratch/frztrunk/PREREG.md).")
    ap.add_argument("--freeze-expect", default="",
                    help="Pre-registered partition sizes, e.g. "
                         "'backbone=180,trunkrest=59,readout=9'; the run dies "
                         "if the built model's partition does not match. The "
                         "freeze is silent when it is wrong, so the count is "
                         "asserted rather than printed.")
    # scratch/truebatch/PREREG.md. Both default to argparse.SUPPRESS: when not
    # passed, the attribute does not exist, so vars(args) / the [args] line /
    # ck["args"] of every pre-existing arm are byte-identical (the per-arm
    # args gates compare key sets). Read them only via getattr(args, k, dflt).
    ap.add_argument("--bn-valid-only", action="store_true",
                    default=argparse.SUPPRESS,
                    help="truebatch: run the backbone only on real (mask!=0) "
                         "series slots so padded zero series never enter "
                         "BatchNorm batch statistics; padded rows get a zero "
                         "feature. Model-affecting (ckpt_build resolves it).")
    # scratch/seqadapt/PREREG.md: argparse.SUPPRESS (truebatch convention) -
    # absent from vars(args)/[args]/ck["args"] unless passed. Read via getattr.
    ap.add_argument("--seq-adapt", type=int, default=argparse.SUPPRESS,
                    help="seqadapt: rank R>0 turns on per-series-type FiLM "
                         "scale/shift + rank-R type-specific residual adapters "
                         "on every ConvNeXt stage output (identity init, live "
                         "gradient). Model-affecting (ckpt_build resolves it). "
                         "Not ported to the Kaggle kernel.")
    # scratch/decouple_20261009/PREREG.md (CONDRES-R+). argparse.SUPPRESS
    # defaults: absent from vars(args) unless passed. Train-only (no
    # parameter, buffer or state-dict key; serve forward unchanged).
    ap.add_argument("--condres-w", type=float, default=argparse.SUPPRESS,
                    help="CONDRES-R+: weight of the T<-C conditional-"
                         "independence penalty on report-T-negative cells.")
    ap.add_argument("--condres-pairs", type=str, default=argparse.SUPPRESS,
                    help="';'-separated 'T<C' finding-name pairs.")
    ap.add_argument("--condres-thr", type=float, default=argparse.SUPPRESS,
                    help="T-negative cell = target <= thr (default .15).")
    ap.add_argument("--condres-bank", type=int, default=argparse.SUPPRESS,
                    help="FIFO bank size per pair (default 256).")
    ap.add_argument("--lr-sched", choices=["onecycle", "cosine"],
                    default=argparse.SUPPRESS,
                    help="truebatch: 'cosine' = linear warmup from max_lr/25 "
                         "over --warmup-pct of steps, then cosine to 0, no "
                         "momentum cycling. Absent = OneCycle (production).")
    # scratch/brierfork/PREREG.md. All default to argparse.SUPPRESS (the
    # truebatch convention): when not passed the attribute does not exist, so
    # vars(args) / [args] / ck["args"] of every other run are byte-identical.
    # Read only via getattr(args, k, dflt).
    ap.add_argument("--fork-save-epoch", type=int, default=argparse.SUPPRESS,
                    help="brierfork: after epoch N-1 finishes (all epoch-end "
                         "evals/saves done, raw weights restored) write "
                         "fold{F}_fork_ep{N}.pt holding the FULL training "
                         "state: model, optimizer, scheduler, EMA (+extras) "
                         "shadow and step count, GradScaler, opt-step count, "
                         "best-trackers and python/numpy/torch/cuda RNG.")
    ap.add_argument("--fork-resume", default=argparse.SUPPRESS,
                    help="brierfork: restore a --fork-save-epoch state and "
                         "continue at its next epoch. Every arg except the "
                         "fork/report-loss/out/stop/workers keys must equal "
                         "the prefix run's (refused otherwise). Best-trackers "
                         "are RESET so fold{F}_ema.pt is the arm's own best "
                         "over the continuation; the terminal EMA is also "
                         "written as fold{F}_fork_terminal_ema.pt.")
    ap.add_argument("--report-loss", choices=["bce", "brier", "scaled"],
                    default=argparse.SUPPRESS,
                    help="brierfork: loss on REPORT rows (studies not in the "
                         "co-train pool) for the main study logits AND the "
                         "per-series aux head; OAI rows keep BCE. bce = the "
                         "original code path untouched; brier = 2(p-y)^2; "
                         "scaled = --report-loss-scale * BCE.")
    ap.add_argument("--report-loss-scale", type=float,
                    default=argparse.SUPPRESS,
                    help="brierfork: scalar c for --report-loss scaled, fixed "
                         "from the --fork-measure-scale read at the fork.")
    ap.add_argument("--fork-measure-scale", type=int,
                    default=argparse.SUPPRESS,
                    help="brierfork: with --fork-resume, run K training micro-"
                         "batches WITHOUT optimizer steps, measure the param-"
                         "gradient norm of the report-row loss under BCE and "
                         "under Brier (main + aux_w*aux), write "
                         "fork_scale.json to --out and exit.")
    ap.add_argument("--sched-advance", type=float, default=0.0,
                    help="Advance the OneCycle schedule by this many EPOCHS' "
                         "worth of optimizer steps before training starts, "
                         "i.e. resume the source run's LR schedule at the "
                         "sub-epoch position its checkpoint was taken from. "
                         "The loop still runs --epochs epochs; the existing "
                         "`sched.last_epoch < steps-1` guard holds the LR at "
                         "the schedule floor for the overhanging tail.")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.2)
    ap.add_argument("--cell-attn", type=int, default=0,
                    help="pool the backbone feature map to a KxK grid and let "
                         "the finding queries attend over (series x slice x "
                         "cell). 0 = off. Zero-init gated: step 0 is identical.")
    ap.add_argument("--comp-pair", default="",
                    help="within-knee COMPARTMENT ordering constraints "
                         "{uid: [[col_lo, col_hi], ...]}. Penalises z[col_lo] > "
                         "z[col_hi] inside one study, so the gradient depends on "
                         "the OTHER column's logit and CAN reorder studies - "
                         "unlike the fixed-bank ranking loss, which was monotone "
                         "and measured +0.00005.")
    ap.add_argument("--comp-pair-w", type=float, default=0.2)
    ap.add_argument("--cell-slice-bands", type=int, default=3,
                    help="through-plane position for cell tokens, as B bands "
                         "(medial/centre/lateral on sagittal). 0 = off.")
    ap.add_argument("--hard-neg", default="",
                    help="FINDING:GATE:MULT[,...] - multiply FINDING's loss "
                         "weight by MULT on studies where GATE is positive, "
                         "renormalised so the column's TOTAL weight is "
                         "unchanged and only its distribution moves. For "
                         "Fracture:Contusion:3.0 - 20.5%% of Fracture's ranking "
                         "pairs are against an oedema-positive non-fracture, "
                         "and that subgroup decides the column.")
    ap.add_argument("--osseous-factor", action="store_true",
                    help="factorize the trauma pair: head rows 10/11 become "
                         "z_E (oedema) and z_L (line|oedema); Contusion = "
                         "E*(1-L), Fracture = E*L, emitted as logits so every "
                         "consumer downstream is unchanged. Parameter-free - "
                         "checkpoint consumers must read it from args, it is "
                         "invisible to state-dict sniffing.")
    ap.add_argument("--frac-neg-flip", action="store_true",
                    help="flip Fracture's negative-side target order: report "
                         "denial -> 0.09, silence -> 0.05, competition uids "
                         "only. Measured anti-calibration fix (miner find #1).")
    ap.add_argument("--oa-gate", action="store_true",
                    help="compartment-gated OA triad: rows 4/5/6 become "
                         "P(i) = sigma(max(z4,z5,z6)) * sigma(z_i), emitted "
                         "as logits. Parameter-free like --osseous-factor; "
                         "consumers must read it from checkpoint args.")
    ap.add_argument("--gem-pool", type=float, default=1.0,
                    help="generalised-mean exponent for the within-slice spatial "
                         "pool. 1.0 = the current global average (bit-identical "
                         "control), higher emphasises peak response. Attacks "
                         "DILUTION - a 1cm lesion is ~1 of 144 cells at the final "
                         "stage - which is a different axis from --cell-attn's "
                         "ADDRESSING. Signed, so it does not discard the ~half of "
                         "post-LayerNorm features that are negative. Try 3.0.")
    ap.add_argument("--gate-init", type=float, default=0.1,
                    help="initial value of the flat/cell attention gate. 0.0 "
                         "gives exact parity but the gated projection then gets "
                         "no gradient and the gate collapses - measured on "
                         "flatattn. 0.1 keeps the branch trainable.")
    ap.add_argument("--cell-stage", type=int, default=-1,
                    help="which backbone stage the cell tokens come from. "
                         "-1 = final (12x12 at 384px, 12.5mm cells); 2 = the "
                         "24x24 map (6.25mm cells).")
    ap.add_argument("--flat-attn", action="store_true",
                    help="per-finding queries also cross-attend over all S*Z "
                         "slice tokens, not just the S pooled series vectors. "
                         "Zero-init gated residual: step 0 is bit-identical.")
    ap.add_argument("--flat-queries", type=str, default="",
                    help="comma list of FINDINGS names whose queries get the "
                         "flat-attn residual; empty = all 12 (unchanged). "
                         "e.g. 'Contusion' for the flatattn2 single-finding arm.")
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--no-decay-norm-bias", action="store_true",
                    help="rb arm (scratch/rb): weight_decay 0 for every 1-D "
                         "parameter (norm scales/biases, LayerScale gammas, "
                         "biases) in backbone and head; two extra param groups "
                         "at the group's own LR. Default off = production "
                         "(every tensor decays at --wd). Byte-identical when off.")
    ap.add_argument("--warmup-pct", type=float, default=0.1,
                    help="OneCycle pct_start (warmup fraction of total steps). "
                         "0.1 = production; rb arm uses 0.3.")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="grad-norm clip threshold; <=0 disables clipping "
                         "(the norm is still measured and logged)")
    ap.add_argument("--clip-groups", action="store_true",
                    help="scratch/qwenvit: clip backbone.* and all other params "
                         "SEPARATELY, each to --clip (qwenvlm v2 lesson: one "
                         "global clip lets a ViT whose grad norm swings 0.02-250 "
                         "starve the head). Off (default) = the single global "
                         "clip, byte-identical. Training-only, not a ckpt key.")
    ap.add_argument("--n-series", type=int, default=4)
    ap.add_argument("--n-slices", type=int, default=16)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--aux-w", type=float, default=0.3)
    ap.add_argument("--aux-visibility-mask", action="store_true",
                    help="auxvis arm (scratch/auxvis): mask the per-series "
                         "aux BCE to the (series, finding) cells where the "
                         "finding is assessable on that series' plane and "
                         "sequence (src/aux_visibility.py); masked cells "
                         "contribute zero loss and leave the mean's "
                         "denominator. Default off = production broadcast. "
                         "Byte-identical when off.")
    ap.add_argument("--rung-heads", default="",
                    help="rubricrung (scratch/rubricrung_20261010/DESIGN.md): "
                         "TRAIN-ONLY per-rung evidence rows, spec "
                         "'f:k,k;f:k' (finding INDEX : report rung k in 1..3). "
                         "Each row is a copy of ev_head row f trained on the "
                         "indicator [report grade >= k] (ladder ends lo/hi "
                         "from this run's prob_table; silent = lo for k>=2, "
                         "excluded for k=1; only RSNA report rows with "
                         "'probs'+'evidence' keys, so gold and OAI rows carry "
                         "no rung loss). Empty (default) = nothing changes.")
    ap.add_argument("--rung-w", type=float, default=1.0,
                    help="weight of the rubricrung loss (mean over rung cells)")
    ap.add_argument("--evdrop", action="store_true",
                    help="evdrop arm (scratch/evdrop/PREREG.md): TRAIN-ONLY "
                         "dominant-evidence dropout. On a fraction "
                         "--evdrop-p of forwards a SECOND per-series pool is "
                         "built from the same slice tokens with the single "
                         "top-evidence slice of every (study, series, FOCAL "
                         "finding) triple removed before the LSE; the study "
                         "loss on that masked view is added to the full-view "
                         "loss over the seven FOCAL columns only. Needs "
                         "--evidence-mil. No new parameter, no state-dict "
                         "change, identity at eval/serve, no kernel port.")
    ap.add_argument("--evdrop-p", type=float, default=0.5,
                    help="probability per forward that the masked view is "
                         "built and its loss added (default 0.5, the value "
                         "frozen in scratch/evdrop/PREREG.md).")
    # --- PAT: patching-based counterfactual training (scratch/shortcutarms) -
    ap.add_argument("--pat", action="store_true",
                    help="PAT arm (scratch/shortcutarms/PREREG.md §PAT, "
                         "Xie et al. ICML 2024): TRAIN-ONLY. Every training "
                         "forward also re-encodes each series as --pat-grid "
                         "contiguous slot SLABS (patches), reads them with the "
                         "same ev_head/V1 pool, aggregates the patch logits by "
                         "a per-finding softmax over patches (pivot), and adds "
                         "--pat-w x soft-BCE of that aggregate vs the study "
                         "target on the 11 non-Synovitis columns. Needs "
                         "--evidence-mil. No parameter, no state-dict change, "
                         "identity at eval/serve, no kernel port.")
    ap.add_argument("--pat-grid", type=int, default=2,
                    help="slabs per series (default 2 = the reference n_grid).")
    ap.add_argument("--pat-w", type=float, default=1.0,
                    help="patch-loss weight (default 1.0 = the reference's "
                         "equal-weight sum).")
    # --- CFCONSIST: counterfactual compartment-mask consistency ------------
    # scratch/cfconsist/PREREG.md (round-5 critic M-2 repaired). TRAIN-ONLY:
    # a second forward on a copy of the input whose MEDIAL (or lateral)
    # half-frame is zeroed on every present Coronal/Axial series (stype >= 2;
    # sagittal untouched - its medial-lateral axis is the slice stack), in the
    # canonical frame (data.py mirrors R knees so medial = image cols < W/2,
    # sign established by scratch/ideas_latoa S_M and anatroi landmarks
    # medial_is_lower_col). Penalty on the TARGET logit change, added to the
    # loss. No parameter/buffer/state-dict key; serve forward unchanged; no
    # kernel port (ckpt_build TRAIN_ONLY_FLAGS).
    ap.add_argument("--cf-consist", type=float, default=0.0,
                    help="weight of the counterfactual-mask consistency term "
                         "(0 = off, byte-identical training).")
    ap.add_argument("--cf-mask", default="medial", choices=["medial", "lateral"],
                    help="half-frame zeroed on Coronal/Axial series.")
    ap.add_argument("--cf-target", default="LatOA",
                    help="finding whose logit must not change under the mask "
                         "(LatOA|MedOA|PFOA or a FINDINGS name).")
    ap.add_argument("--cf-mode", default="drop", choices=["drop", "sym"],
                    help="z_orig is DETACHED (sg). drop: relu(z_orig - z_mask)^2 (a FALL of "
                         "the target logit when the other compartment is "
                         "removed); sym: (z_mask - z_orig)^2 (invariance).")
    ap.add_argument("--cf-converse", action="store_true",
                    help="also add the converse pair (opposite half masked, "
                         "MedOA<->LatOA target) at the same weight.")
    ap.add_argument("--cf-prob", type=float, default=1.0,
                    help="probability per training micro-step that the "
                         "counterfactual forward is run (default 1.0).")
    ap.add_argument("--darc-local", action="store_true",
                    help="DARC-local arm (scratch/shortcutarms/PREREG.md §3, "
                         "HELD): per-finding total-direct-effect evidence - "
                         "each non-Synovitis ev_head row is read with every "
                         "companion row direction projected out (stop-grad). "
                         "SERVE-PATH, zero parameters; needs --evidence-mil; "
                         "shipping needs a kernel port or export-time fold.")
    # --- ev-adapter / conditional evidence FORMATION (scratch/evform) -----
    ap.add_argument("--ev-adapter", default="off", choices=["off", "mlp", "moe"],
                    help="evform arm (scratch/evform/PREREG.md): a residual, "
                         "soft-routed low-rank adapter on the 768-d slice "
                         "token, inserted immediately BEFORE the per-slice "
                         "evidence readout: h~ = h + sum_k pi_k(h) U_k "
                         "phi(V_k LN(h)), pi = softmax(W_route LN(h)). "
                         "'moe' is the arm (K experts), 'mlp' the matched-"
                         "CAPACITY control (one expert, no router, width "
                         "derived to match the moe parameter count). U=0 at "
                         "init so both are function-preserving; V and W_route "
                         "get normal init so the router carries gradient from "
                         "step 0. Needs --evidence-mil. PARAMETER-BEARING "
                         "(ev_adapter_mod.* keys): ck['args'] is the contract, "
                         "src/ckpt_build.py cross-checks both directions and "
                         "the kernel REFUSES it until a port exists.")
    ap.add_argument("--ev-adapter-k", type=int, default=4,
                    help="number of moe experts (default 4, frozen in "
                         "scratch/evform/PREREG.md). Also sets the 'mlp' "
                         "control's matched width.")
    ap.add_argument("--ev-adapter-width", type=int, default=32,
                    help="per-expert bottleneck width (default 32). The 'mlp' "
                         "control derives its own width from (k, width).")
    ap.add_argument("--ev-adapter-lr", type=float, default=0.0,
                    help="absolute LR for the ev_adapter_mod.* parameter "
                         "group. 0 (default) = 5 x --lr: scratch/evwave/"
                         "evR_read/RESULTS.md measured the EV reliability "
                         "parameter never moving off its init at the head LR, "
                         "so a fresh adapter in front of the same readout is "
                         "given its own faster group rather than inheriting "
                         "that schedule.")
    # --- WITNESS-EV (scratch/witness/PREREG.md) ---------------------------
    ap.add_argument("--witness-targets", default="",
                    help="WITNESS-EV arm (scratch/witness/PREREG.md): npz of "
                         "per-series evidence from the honest-OOF EV parent "
                         "members (the member that HELD OUT each study), keyed "
                         "by study UID x series UID. TRAIN-ONLY: adds a KL "
                         "from the teacher's across-series evidence softmax to "
                         "the student's, on the 11 non-Synovitis findings. "
                         "Nothing changes at inference; the flag is registered "
                         "in src/ckpt_build.py:TRAIN_ONLY_FLAGS. Default '' = "
                         "off, byte-identical to every pre-flag arm.")
    ap.add_argument("--witness-beta", type=float, default=0.5,
                    help="weight of the witness KL in the total loss. The "
                         "arm's FROZEN value is 2 (scratch/witness/PREREG.md "
                         "section 4 dose rule: beta*L_witness ~ 0.10 x the "
                         "soft-BCE at initialisation), passed explicitly by "
                         "scratch/witness/launch.sh; this argparse default is "
                         "not the arm's dose.")
    ap.add_argument("--witness-tau", type=float, default=1.0,
                    help="ONE temperature, applied to the teacher and the "
                         "student evidence alike before the across-series "
                         "softmax (frozen at 1.0).")
    ap.add_argument("--witness-gate", default="entropy_conf",
                    choices=["none", "entropy", "entropy_conf"],
                    help="per-cell gate weight c. 'entropy_conf' (FROZEN) uses "
                         "c = (1 - H(w_T)/log(S_present)) * t, t the blended "
                         "study target for the cell: a diffuse witness "
                         "contributes almost nothing, and a concentrated "
                         "witness on a NEGATIVE cell is noise, so the term is "
                         "weighted onto positive-evidence cells. 'entropy' "
                         "drops the confidence factor, 'none' uses c = 1.")
    # --- SLICE-WITNESS (scratch/slicewit/PREREG.md) -----------------------
    ap.add_argument("--slicewit-targets", default="",
                    help="slicewit arm (scratch/slicewit/PREREG.md): npz of the "
                         "honest-OOF EV members' PER-SLICE evidence keyed by "
                         "(study, series, native slice). Adds the training-only "
                         "term beta * gate-weighted-mean KL(v_T || v_S), v = "
                         "softmax over the shared native slices of a series of "
                         "(slice evidence / tau), gate entropy_conf, 11 non-"
                         "Synovitis columns. Empty = off (byte-identical).")
    ap.add_argument("--slicewit-beta", type=float, default=1.0,
                    help="weight of the slice-witness KL; the arm's value is "
                         "fixed by the PREREG dose rule (scratch/slicewit/"
                         "dose.json), not by this default")
    ap.add_argument("--slicewit-tau", type=float, default=1.0,
                    help="one softmax temperature for teacher AND student")
    ap.add_argument("--auxbag", "--aux-bag", dest="auxbag",
                    action="store_true",
                    help="auxbag arm (scratch/auxbag/PREREG.md): replace the "
                         "broadcast per-series aux BCE with ONE bag "
                         "likelihood per finding over the valid series, "
                         "Q_c = 1 - prod_s (1 - q_sc), "
                         "L_aux,c = BCE(t_c, Q_c) / S_valid. Masked series "
                         "contribute log(1-q)=0 and are exactly absent from "
                         "the product. aux_w, the ladder target, the study "
                         "loss and osseous factorisation are unchanged; the "
                         "per-series aux bias is initialised to "
                         "logit(1-(1-pi_c)^(1/5)) so the initial bag "
                         "probability is not ~1. TRAIN-ONLY (the aux head is "
                         "not consumed at serve time). Mutually exclusive "
                         "with --aux-visibility-mask and with label repair. "
                         "Byte-identical when off.")
    ap.add_argument("--trauma-head", action="store_true", default=argparse.SUPPRESS,
                    help="scratch/traumahead: non-exclusive trauma pair P(C)=E*K, P(F)=E*L with an independent "
                         "contusion factor K (replaces the ossfact P(C)=E(1-L) on the evmil path). SUPPRESS: absent "
                         "from [args] unless passed.")
    ap.add_argument("--evidence-mil", action="store_true",
                    help="evmil arm (scratch/evmil/PREREG.md): REPLACE the "
                         "attn-pooled series token -> study transformer -> 12 "
                         "cross-attention queries with pathology-specific "
                         "evidence accumulation (per-slice per-finding "
                         "evidence, LSE-over-slices for focal findings / mean "
                         "for diffuse, then a learned per-finding reliability "
                         "weighting over the present series initialised from "
                         "a clinical plane/sequence prior). study_enc and its "
                         "queries are NOT constructed; the per-series pooled "
                         "evidence replaces the aux head and is supervised by "
                         "the SAME broadcast aux BCE at --aux-w. Default off "
                         "= byte-identical to before this flag existed. "
                         "Stored in ck['args']; consumers must rebuild with "
                         "it (the weights also carry ev_R / ev_head.*).")
    ap.add_argument("--rubric-pool", action="store_true",
                    help="rubpool arm (scratch/rubpool/PREREG.md): REPLACE the "
                         "attn-pooled series token -> study transformer -> 12 "
                         "cross-attention queries with a FIXED, rubric-shaped "
                         "pooling of per-slice per-finding evidence: meniscus "
                         "= mean of the top-2 slices then max over series "
                         "('reaches the surface on >=2 images'); OA/Effusion/"
                         "Baker's = log-sum-exp soft extent then max over "
                         "series; ACL/MCL/Fracture/Contusion = max then max; "
                         "Synovitis = mean then mean. Nothing in the pooling "
                         "is learned (no attention, no temperature, no "
                         "reliability table); the head's only parameters are "
                         "rub_head.* and rub_b. The per-series aux head is "
                         "UNCHANGED (the difference from --evidence-mil). "
                         "Default off = byte-identical to before this flag "
                         "existed. Stored in ck['args']; consumers must "
                         "rebuild with it (weights also carry rub_head.*).")
    ap.add_argument("--evmil-prior", default="clinical",
                    choices=["clinical", "fracbone", "confirm"],
                    help="which clinical reliability prior INITIALISES ev_R "
                         "(scratch/fracprior/PREREG.md, "
                         "scratch/priorpartner/PREREG.md). ev_R is a saved "
                         "nn.Parameter that was measured never to move off its "
                         "init, so the prior IS the model's series routing. "
                         "clinical = the shipped prior (default; R is "
                         "byte-identical to every pre-existing arm). fracbone "
                         "= Fracture pointed at NON-fat-sat series (cortical "
                         "break is a T1/PD sign), Contusion unchanged. confirm "
                         "= confirmatory-plane weighting for ACL/MCL/both "
                         "menisci/Baker's, Fracture and Contusion unchanged. "
                         "Init only: no effect without --evidence-mil, and "
                         "serve-inert (rebuilds load the trained ev_R). Stored "
                         "in ck['args'] as the record of which prior ran.")
    ap.add_argument("--evmil-tau-init", type=float, default=1.0,
                    help="init of the per-finding LSE temperature ev_tau on "
                         "the FOCAL-7 findings (scratch/evpool/PREREG.md). "
                         "tau -> 0 is max-pooling (one slice decides), tau -> "
                         "inf is the mean; the forward clamps at 0.2. ev_tau "
                         "is a saved nn.Parameter that was MEASURED not to "
                         "move off its init (0.92-1.01 on the served seed-1 "
                         "checkpoints), so this init IS the aggregator. Init "
                         "only: no effect without --evidence-mil, and "
                         "serve-inert (rebuilds and the kernel load the "
                         "trained ev_tau). Default 1.0 = byte-identical to "
                         "every pre-flag arm.")
    ap.add_argument("--evmil-dif-init", default="mean",
                    choices=["mean", "quantile", "max"],
                    help="which member of the DIFFUSE-5 slice mixture "
                         "(mean / quantile-0.8 / max) the softmax ev_mix "
                         "starts dominated by (scratch/evpool/PREREG.md). "
                         "mean = logits [2,0,0], the shipped init; quantile = "
                         "[0,2,0]; max = [0,0,2]. ev_mix is a saved "
                         "nn.Parameter measured not to move off its init "
                         "(softmax stays 0.78/0.11/0.11), so this init IS the "
                         "diffuse aggregator. Init only, serve-inert; stored "
                         "in ck['args'] as the record of which flavour ran.")
    ap.add_argument("--evidence-mil-v2", default="off",
                    choices=["off", "evw", "mean"],
                    help="evmil V2a arm (scratch/evmil_v2/PREREG.md): add a 32d "
                         "MORPHOLOGY RESIDUAL to the focal findings' per-series "
                         "evidence, read at V1's OWN evidence locations. "
                         "z=GELU(LN(W_f h_t)) 768->32 per focal finding; "
                         "evw = alpha is a DETACHED softmax of V1's smoothed "
                         "evidence over slices / V1's tau; mean = CONTROL, "
                         "uniform alpha (same parameters, no localisation). "
                         "r_f = 32->16->1, added as gamma_f * r_f with gamma "
                         "ZERO-init, so step 0 is bit-identical to V1. Requires "
                         "--evidence-mil. New parameters take the HEAD LR "
                         "group. Stored in ck['args']; evw and mean are "
                         "structurally identical, so the mode CANNOT be sniffed "
                         "from the weights and every consumer must read args "
                         "(src/ckpt_build.py refuses a checkpoint whose ev2_* "
                         "weights and args disagree).")
    ap.add_argument("--syno-specialist", action="store_true",
                    help="synospec arm (scratch/synov2/PREREG.md Amendment 1): "
                         "add a BUNDLED Synovitis-only residual to V1's STUDY "
                         "Synovitis logit - gB*r_B (9 burden/extent statistics "
                         "of the per-slice Synovitis evidence, reliability-"
                         "weighted over series) + gC*r_C (V2a's 32d morphology "
                         "read on the Synovitis row, detached alpha) + gD*r_D "
                         "(stopgrad of the V1 study Effusion logit). The other "
                         "eleven logits and the per-series evidence out[1] are "
                         "bit-identical to V1. Requires --evidence-mil; refuses "
                         "--evidence-mil-v2 and --bypass-study-transformer.")
    ap.add_argument("--syno-gamma-init", type=float, default=0.01,
                    help="init of --syno-specialist's three residual gains "
                         "[gB, gC, gD]. 0.0 makes the arm bit-identical to V1 "
                         "at step 0 and is the equivalence gate's setting, but "
                         "it is NOT the training default: the V2a T4 dynamics "
                         "probe measured the CLAUDE.md zero-init-gate collapse "
                         "at 0, so 0.01 ships (recorded in ck['args']).")
    ap.add_argument("--ev2-gamma-init", type=float, default=0.01,
                    help="init of --evidence-mil-v2's per-finding residual gain "
                         "gamma_f. 0.0 makes the arm BIT-IDENTICAL to V1 at step "
                         "0; the T4 dynamics probe measured that a zero init "
                         "COLLAPSES (|gamma| random-walks under 1e-3 while "
                         "AdamW's decoupled decay pulls it back), which is the "
                         "CLAUDE.md zero-init-gate tripwire, so the PREREG's "
                         "named fallback 0.01 is the default "
                         "(scratch/evmil_v2/t4probe/RESULTS.md).")
    ap.add_argument("--pair-rank", type=float, default=0.0,
                    help="weight of the longitudinal pair loss: for OAI knees "
                         "scanned at V00 and 96m, softplus(z_earlier - "
                         "z_later) on the three OA columns. The 96m studies "
                         "enter training ONLY through this loss (no labels, "
                         "not in `studies`). 0 = off.")
    ap.add_argument("--pair-map", default="data/processed/oai_pairs.json",
                    help="{v00_uid: 96m_uid} map for --pair-rank")
    ap.add_argument("--pair-every", type=int, default=4,
                    help="apply the pair loss every N optimizer micro-steps "
                         "(each application costs two extra forwards)")
    ap.add_argument("--bilat-rank", type=float, default=0.0,
                    help="weight of the BILATERAL same-subject same-visit "
                         "L/R ranking loss (scratch/oaibilat/PREREG.md): for "
                         "each usable OAI left/right pair and each of the "
                         "three challenge OA compartments where the two "
                         "knees' depth-aware rubric severities differ "
                         "(scratch/latoarescue/RUBRIC_MAP.md), a softplus "
                         "soft-margin (margin 0) ranking term pushes the "
                         "worse knee's OA logit above the better knee's. "
                         "Ties/masked are excluded at build time. Both "
                         "knees are forwarded in the SAME micro-step (same "
                         "optimizer step; per-side exact backwards, since "
                         "two study graphs OOM one L40S) and carry NO "
                         "labels - the challenge loss is never "
                         "double-counted. Cross-sectional L-vs-R, so "
                         "person/site/scanner nuisance cancels exactly; NOT "
                         "the longitudinal --pair-rank object. 0 = off, "
                         "byte-path-inert.")
    ap.add_argument("--bilat-pairs",
                    default="data/processed/oai_bilateral_pairs_strata.json",
                    help="per-pair per-compartment discordance strata json "
                         "from scratch/oaibilat/build_strata.py")
    ap.add_argument("--bilat-every", type=int, default=8,
                    help="apply the bilateral pair loss every N micro-steps "
                         "(8 = exactly once per bs1/accum8 optimizer step; "
                         "each application costs three extra forwards + two "
                         "extra backwards, ~2.3 micro-steps of compute)")
    # scratch/noisystud/PREREG_dose.md (Addendum D): step-matched auxiliary
    # pool stream. The MAIN loader (and therefore steps/epoch, OneCycle
    # total_steps, EMA and snapshot indices) is untouched; every
    # --aux-pool-every main micro-batches one extra study from a SEPARATE pool
    # is forwarded and trained on its teacher probabilities at weight
    # --aux-pool-w (all 12 cells, coverage 1), /accum, into the same optimizer
    # step. Off (empty --aux-pool-teacher) = no dataset, no loader, no RNG
    # draw, byte-path-identical to every prior arm.
    ap.add_argument("--aux-pool-teacher", default="",
                    help="NPZ (uids, targets, findings; --teacher-targets "
                         "format) of teacher probabilities for the auxiliary "
                         "pool. Must be disjoint from gold-58 and from every "
                         "main-pool (RSNA/OAI/val) study. Empty = OFF.")
    ap.add_argument("--aux-pool-vol-root", default="",
                    help="volume root holding the auxiliary pool studies")
    ap.add_argument("--aux-pool-series-csv", default="",
                    help="series index covering the auxiliary pool studies")
    ap.add_argument("--aux-pool-laterality-map", default="",
                    help="laterality map covering the auxiliary pool studies")
    ap.add_argument("--aux-pool-every", type=int, default=1,
                    help="one auxiliary-pool study every K main micro-batches")
    ap.add_argument("--aux-pool-w", type=float, default=0.0,
                    help="weight on the auxiliary-pool soft-BCE (teacher "
                         "targets), applied before the /accum division")
    ap.add_argument("--main-order-hash", type=int, default=0,
                    help="diagnostic only: print a sha1 of the first N main "
                         "training-loader study ids of epoch 0 (reads "
                         "b['study'] strings; no RNG, no tensor op). 0 = off.")
    ap.add_argument("--ret-anchor-w", type=float, default=0.0,
                    help="scratch/retanchor/PREREG.md: within-run "
                         "self-anchored function preservation. At the END of "
                         "epoch --ret-anchor-epoch the run saves its EMA "
                         "weights to fold{N}_anchor_ema.pt and keeps a "
                         "frozen in-memory copy (eval mode, no grad); from "
                         "the NEXT epoch onward EVERY training batch "
                         "(challenge and cotrain rows alike) adds "
                         "w * soft-BCE(live 12 challenge-finding logits, "
                         "anchor probabilities) to the loss. Label-free: "
                         "label losses stay untouched at full weight; "
                         "aux/series/report heads are excluded (out[0] "
                         "only). Train-time only - the terminal artifact is "
                         "the standard fold{N}_ema.pt and serve cost is "
                         "zero. 0 = OFF and byte-path-identical to the "
                         "incumbent (same standard as --moaks-aux / "
                         "--bilat-rank).")
    ap.add_argument("--ret-anchor-epoch", type=int, default=5,
                    help="epoch whose END-of-epoch EMA becomes the anchor. "
                         "FROZEN at 5 by the prereg (family median gold "
                         "peak) - exists only so the value is recorded in "
                         "ck['args'], NOT a sweep axis.")
    ap.add_argument("--drop-fold", type=int, default=-1,
                    help="exclude this fold's groups from the TRAIN half "
                         "(nested-CV teacher discipline); val is --fold as "
                         "usual. -1 = off.")
    ap.add_argument("--slice-enc", default="tr", choices=["tr", "lstm", "none"],
                    help="slice-sequence mixer: tr = shipped transformer; "
                         "lstm = bi-LSTM (RSNA-2024 winners' ablation ranked "
                         "it above self-attention); none = field-style 2.5D "
                         "(per-slice features + attention pool only).")
    ap.add_argument("--site-aux", type=float, default=0.0,
                    help="weight of a PLAIN supervised head predicting scanner "
                         "manufacturer from the study vector - the opposite "
                         "sign of --dann (adds site information instead of "
                         "removing it; RSNA-Aneurysm 2nd place used it to "
                         "stabilise cross-scanner training). Same external "
                         "head; checkpoints unchanged. 0 = off.")
    ap.add_argument("--dann", type=float, default=0.0,
                    help="weight of the domain-adversarial (site) loss: a "
                         "gradient-reversal head predicts scanner manufacturer "
                         "from the study vector, pushing the trunk toward "
                         "site-invariant features. Head lives outside the "
                         "model; checkpoints are unchanged. 0 = off.")
    ap.add_argument("--aucm", type=float, default=0.0,
                    help="weight of the AUC-margin (Yuan et al., ICCV 2021) "
                         "auxiliary loss on --aucm-findings. Saddle-point "
                         "surrogate on sigmoid probs with SOFT pos/neg masses "
                         "w+ = relu(2(y-.5)), w- = relu(2(.5-y)) so the "
                         "ordinal ladder is never binarised (the gate defect "
                         "that killed --rank-w). Positives are pulled toward "
                         "a+alpha and negatives toward b-alpha, so pos/neg "
                         "pairs CAN reorder - unlike the fixed-bank loss. "
                         "State (a, b, alpha) lives outside the model; "
                         "checkpoints unchanged. 0 = off.")
    ap.add_argument("--aucm-findings", default="MCL,Fracture,Lateral OA",
                    help="comma list of findings the AUCM term applies to. "
                         "Default = low-prevalence heads, where the published "
                         "dose-response concentrates - and the untreated nine "
                         "are the internal placebo.")
    ap.add_argument("--aucm-margin", type=float, default=1.0)
    ap.add_argument("--aucm-lr", type=float, default=0.01,
                    help="lr for (a, b) descent and the alpha dual ascent")
    ap.add_argument("--bagauc-w", type=float, default=0.0,
                    help="weight of the grade-BAG pairwise AUC loss "
                         "(scratch/bagauc/PREREG.md). 0 (default) = off, "
                         "byte-identical to plain soft-BCE. From epoch "
                         "--bagauc-start-ep this REPLACES (not adds to) "
                         "soft-BCE for every finding: bags are the ordinal "
                         "grade classes themselves (silent=-1, 0..3, never "
                         "binarised), pi(bag) is read from the SAME "
                         "PRIOR_TABLE / --silent-calib table the soft "
                         "targets are built from, and the loss is the mean "
                         "over genuine cross-bag pairs of |pi_a-pi_b| * "
                         "softplus(-(z_a-z_b)) with BOTH z_a, z_b live in "
                         "the CURRENT accumulation window (never a detached "
                         "bank - that formulation is the killed --rank-w "
                         "monotone-map trap, see its help string).")
    ap.add_argument("--bagauc-start-ep", type=int, default=2,
                    help="first epoch (0-indexed) at which --bagauc-w "
                         "replaces soft-BCE; epochs before this train on "
                         "plain soft-BCE so representations form on the "
                         "calibrated ladder first (PREREG rationale: "
                         "Synovitis silent-block ordering is still >=0.835 "
                         "at ep1.7, ceiling-attractor pull dominates after "
                         "~ep2).")
    ap.add_argument("--bagauc-margin-form", default="logistic",
                    choices=["logistic"],
                    help="pairwise surrogate form. Only 'logistic' "
                         "(softplus(-(z_a-z_b))) is implemented; the flag "
                         "exists so a future form change is visible in "
                         "[args] rather than silent.")
    ap.add_argument("--bagauc-gradcache", action="store_true",
                    help="GradCache two-pass (Gao et al. 2021) evaluation of "
                         "--bagauc-w, MATHEMATICALLY EXACT vs the default "
                         "retain_graph path (never an approximation) but "
                         "with only ONE backbone graph resident at a time "
                         "instead of --accum. Fixes the OOM the default path "
                         "hits the instant bag-AUC activates (jobs "
                         "59322185, 59324022 - 44.33/44.39 GiB on an L40S, "
                         "with AND without --grad-checkpoint): P = the "
                         "window's bag-AUC term depends on theta only "
                         "through the window's (N,12) logits Z, so dL/dtheta "
                         "= sum_i d(local_i)/dtheta + sum_i (dP/dz_i)(dz_i/"
                         "dtheta). Three passes per window: (A) no_grad "
                         "forward of every study, collecting detached "
                         "logits; (B) a tiny detached graph running the "
                         "existing _bagauc_window_loss on those logits to "
                         "get G = dP/dZ (no backbone involved); (C) re-"
                         "forward each study WITH grad, RNG-restored to "
                         "reproduce pass A's forward exactly (asserted), add "
                         "the constant surrogate bagauc_w*(logits_i . G_i) "
                         "to that study's local loss, and backward "
                         "immediately so the graph frees before the next "
                         "study. Default OFF: the retain_graph path is "
                         "untouched byte-for-byte when this flag is not "
                         "given. scratch/bagauc/PREREG.md gradcache "
                         "addendum, 2026-08-20; unit test "
                         "scratch/bagauc/test_gradcache.py.")
    ap.add_argument("--pair-route-reg", type=float, default=0.0,
                    help="weight on the total-variation distance between the "
                         "routing attention of anatomically paired findings "
                         "(Med/Lat Meniscus, Med/Lat OA). Both compartments sit "
                         "on the SAME images, yet their queries attend to "
                         "near-disjoint series (TV 0.79/0.57, cosine 0.154) - "
                         "so neither sees all the evidence. This pulls each "
                         "pair's EVIDENCE together while leaving the queries "
                         "and heads free to differ; 0 skips entirely.")
    ap.add_argument("--text-emb", default="data/processed/report_emb_centred.npz",
                    help="report sentence embeddings; '' disables the text loss")
    ap.add_argument("--text-w", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA decay on the weights; 0 disables it. The EMA "
                         "model is scored and saved separately, never mixed "
                         "with the raw one, so enabling it cannot make the "
                         "raw checkpoint worse.")
    ap.add_argument("--rep-gls-sharp", type=float, default=0.0,
                    help="Rep-GLS sharpening magnitude s>=0, applied ONLY to "
                         "cells the report stated CATEGORICALLY (grade 0 or 3, "
                         "detected by exact match against the calibrated "
                         "ladder's end values). Loss becomes (1+s)*BCE - "
                         "s*KL(p||uniform) there, i.e. an anti-regulariser that "
                         "pushes away from 0.5. Our ladder only ever SOFTENS; "
                         "this is the half of generalized label smoothing the "
                         "pipeline has never had, and the published ablation "
                         "attributes +2.58 of +4.51 to it. 0.0 is bit-identical "
                         "to the old loss.")
    ap.add_argument("--query-family", default="none",
                    choices=["none", "clinical", "random"],
                    help="add a learned per-FAMILY vector to each of the twelve "
                         "finding queries. 'clinical' groups them "
                         "ligament/meniscus/OA/inflammation/cystic/bone, so the "
                         "well-measured member of a family can carry routing "
                         "information to the weak one (Fracture->Contusion, "
                         "Effusion->Synovitis, Medial OA->Lateral OA). "
                         "'random' is the PLACEBO: identical parameter count "
                         "and identical family-size profile with the clinical "
                         "structure destroyed. Run both or neither - a "
                         "'clinical' gain with no 'random' control is a "
                         "capacity result wearing a mechanism's clothes.")
    ap.add_argument("--ema-extra-decays", default="",
                    help="comma list of ADDITIONAL decays to carry as parallel "
                         "shadows, e.g. '0.995,0.998'. Each gets its own "
                         "float32 copy, its own per-epoch score and its own "
                         "checkpoint fold{N}_ema_d<tag>.pt; nothing selects "
                         "between them here. Only decays below ~0.998 bind at "
                         "all - the ramp caps the effective decay at ~0.9983 "
                         "for a standard-length run, so 0.999/0.9999/1.0 are "
                         "the SAME shadow (see EMA docstring). Empty (default) "
                         "leaves the run bit-identical to before this flag.")
    ap.add_argument("--ema-extra-start", type=int, default=4,
                    help="first epoch at which the extra shadows are evaluated "
                         "and saved. They still UPDATE from step 0 - this only "
                         "skips the scoring passes over epochs no selector has "
                         "ever picked (selection lands ep7-10), which is where "
                         "the sweep's wall-clock cost would otherwise go.")
    ap.add_argument("--calib", default="fixed",
                    choices=["fixed", "gold", "kfold"],
                    help="grade->probability table. 'fixed' uses a prior and "
                         "touches no gold label (default, keeps GOLD-AUC "
                         "honest); 'gold' fits on all 58 and leaks them into "
                         "the training targets; 'kfold' cross-fits.")
    ap.add_argument("--calib-fold", type=int, default=0)
    ap.add_argument("--calib-folds", type=int, default=4)
    ap.add_argument("--save-early-epoch", type=int, default=1,
                    help="Also save an UNCONDITIONAL checkpoint at this epoch. "
                         "Findings whose labels are unreliable peak early and "
                         "then erode: Synovitis scored 0.84/0.75/0.81 at epoch 0 "
                         "in arms A/B/D and 0.69/0.68/0.67 at epoch 11, while "
                         "ACL went 0.66->0.97, 0.61->0.95, 0.60->0.96. One "
                         "checkpoint cannot hold both. This is an EXTRA file - "
                         "fold{N}_best.pt is unaffected - so it costs disk and "
                         "nothing else. -1 disables.")
    ap.add_argument("--silent-calib", default="none", choices=["none", "gold"],
                    help="'gold' fits P(gold=1 | report silent) PER FINDING, "
                         "shrunk toward the pooled value. Silence means "
                         "different things per finding - Synovitis 0.258 and "
                         "PF OA 0.250 vs 0.000 for ACL/Medial OA - and the -1 "
                         "cell holds the most studies exactly where the pooled "
                         "value is most wrong. Reads gold labels, so it "
                         "reintroduces calibration leakage; the leaderboard has "
                         "to arbitrate, not gold-58.")
    ap.add_argument("--silent-shrink", type=float, default=10.0)
    ap.add_argument("--report-cell-weight", default="",
                    help="scratch/rfsyno: comma list of Finding:weight. Scales "
                         "the soft-BCE weight of the named findings' cells on "
                         "REPORT-derived rows only (rows of --labels that carry "
                         "an 'evidence' key; OAI/cotrain rows carry none and "
                         "are untouched). 'Synovitis:0.0' = the Synovitis head "
                         "is supervised by the MOAKS pool alone (graded AND "
                         "silent report cells removed from ep0). Differs from "
                         "--silent-mute-*, which touches only grade -1 cells. "
                         "Empty (default) = no op, weight tensor untouched.")
    ap.add_argument("--silent-mute-from", type=float, default=-1.0,
                    help="Fractional epoch from which cells whose report grade "
                         "is -1 (EXPLICITLY SILENT) get their soft-BCE weight "
                         "scaled by --silent-mute-w. -1 disables (the default, "
                         "and every pre-existing arm - the weight tensor is "
                         "then untouched). Isolates the one force no prior arm "
                         "did: --mute-after removed a finding's silent AND "
                         "graded supervision together and only for that "
                         "finding; --bagauc-w removed pointwise BCE from "
                         "silent AND graded cells and REPLACED it with a "
                         "cross-bag ranking term. Neither tested the ABSENCE "
                         "of the fabricated individual target on silence. "
                         "P(gold=1|silent) is a POPULATION statement, not an "
                         "observation that THIS knee is 10%% positive. Grade "
                         "-99 (probs-only rows carrying no 'grades' key) is "
                         "NEVER muted - absent is not silent. Prereg: "
                         "scratch/silmute/PREREG.md.")
    ap.add_argument("--silent-mute-findings", default="",
                    help="scratch/silweight: comma list of finding names whose "
                         "silent cells are affected by --silent-mute-from/-w; "
                         "empty (default) = all 12, byte-identical to silmute.")
    ap.add_argument("--silent-mute-w", type=float, default=0.0,
                    help="Weight multiplier applied to silent cells once "
                         "--silent-mute-from is active. 0.0 = full mute (the "
                         "pre-registered arm). A weight rather than a boolean "
                         "so the pre-registered partial-dose follow-up (0.25) "
                         "needs no new code - see the ladder in the prereg. "
                         "Muting deletes 21.8%% of all supervised cells, but "
                         "NON-UNIFORMLY: 65.1%% of Synovitis, 45.9%% of "
                         "Baker's, 5.2%% of Medial Meniscus.")
    ap.add_argument("--wd-from", type=float, default=-1.0,
                    help="Fractional epoch from which the BACKBONE param "
                         "group's weight_decay is multiplied by --wd-scale. "
                         "-1 disables (the default, and every pre-existing "
                         "arm). Motivated by measurement, not by analogy: the "
                         "ep0.30->terminal drift has per-tensor "
                         "cos(dtheta,-theta) median +0.52 to +0.69 in the "
                         "BACKBONE (97-98%% of tensors) but only +0.02 to "
                         "+0.04 in the densely-supervised readout, replicated "
                         "on 3 arms, and it escalates monotonically with "
                         "training (0.10 early -> 0.55 late). AdamW's "
                         "decoupled decay is the only systematic NON-gradient "
                         "term in this optimizer, which is exactly what "
                         "gradown's 'no single-step gradient owns the drift' "
                         "result calls for. Prereg: scratch/wdecay/PREREG.md.")
    ap.add_argument("--wd-target", default="backbone",
                    choices=["backbone", "readout"],
                    help="Which half of the network --wd-from acts on. "
                         "'backbone' is the TREATMENT: that is where the "
                         "shrinkage signature lives (per-tensor "
                         "cos(dtheta,-theta) median +0.52 to +0.69, 97-98%% of "
                         "tensors). 'readout' is the PLACEBO/localisation "
                         "CONTROL: the signature is essentially ABSENT there "
                         "(+0.017 to +0.038, 66-71%%), so the retention "
                         "hypothesis predicts NO effect while the "
                         "'we merely removed regularisation' hypothesis "
                         "predicts a similar one. Running both is what "
                         "separates them.")
    ap.add_argument("--wd-scale", type=float, default=0.0,
                    help="Multiplier on the backbone group's weight decay once "
                         "--wd-from is active. 0.0 = decay off (the "
                         "pre-registered arm). A multiplier rather than a "
                         "boolean so the partial-dose follow-up (0.25) needs "
                         "no new code.")
    ap.add_argument("--gram-escrow-from", type=float, default=-1.0,
                    help="Fractional epoch at which to (a) cache the EMA "
                         "model's per-study trunk representations and (b) "
                         "start penalising drift of each study's SIMILARITY "
                         "PROFILE to a fixed reference set. -1 disables (the "
                         "default, and every pre-existing arm). Soft "
                         "counterpart to --freeze: frztrunk_v1 showed a hard "
                         "trunk freeze costs the late-built signals (Cont|g2 "
                         "0.640 vs its control's 0.804) and 0.13 of gold, so "
                         "the constraint must penalise DRIFT, not pin WEIGHTS. "
                         "Prereg: scratch/gramesc/PREREG.md.")
    ap.add_argument("--gram-escrow-w", type=float, default=0.0,
                    help="lambda on the escrow term. 0.0 disables. Fixed by "
                         "the pre-registered RULE (0.1 x the ratio of gradient "
                         "norms, measured in the smoke), NEVER by an argmax "
                         "sweep - that is the max-of-K hazard that killed the "
                         "MedMen/LatOA epoch gaps.")
    ap.add_argument("--gram-escrow-k", type=int, default=256,
                    help="Size of the fixed reference set the similarity "
                         "profile is taken against, sampled deterministically "
                         "with RandomState(0) from the train pool. K < D=768 "
                         "on purpose: the constraint then binds only the "
                         "projection onto the reference span, leaving the "
                         "rest of the representation free.")
    ap.add_argument("--gram-escrow-probe", type=int, default=0,
                    help="Diagnostic: report ||grad L_standard|| and "
                         "||grad L_gram|| for this many windows after "
                         "activation, then exit(0). Produces the number the "
                         "lambda RULE consumes; never set on a funded arm.")
    ap.add_argument("--coverage", default="",
                    help="npz with uids + coverage (N,12) - per-study, per-"
                         "finding loss weight for co-training on an EXTERNAL "
                         "pool. An OAI study scored with MOAKS supervises "
                         "Synovitis and the compartment-specific items and "
                         "must be silent on the findings MOAKS does not score, "
                         "rather than being taught a fabricated negative. "
                         "Applied to the TRAIN half only.")
    ap.add_argument("--label-noise", type=float, default=0.0,
                    help="B1 dose-response instrument. Destroy the information "
                         "in this FRACTION of training label cells, by replacing "
                         "each selected (study, finding) target with that "
                         "finding's target drawn from another random training "
                         "study. TRAIN HALF ONLY and never a gold-58 study, so "
                         "the val readout and the gold readout both stay clean. "
                         "0.0 (default) = off. Purpose: the claim that 88%% of "
                         "remaining headroom is label-bound rests on a "
                         "CROSS-FINDING correlation (AUC vs label ceiling, "
                         "slope +0.30, CI [+0.132,+0.462], 12 points). This "
                         "flag makes that slope CAUSAL by intervening on label "
                         "quality within a fixed recipe.")
    ap.add_argument("--label-noise-seed", type=int, default=0)
    ap.add_argument("--stage-taps", default="",
                    help="CSV of backbone stage indices (e.g. '1,2,3') to fit "
                         "DETACHED linear probes at, per finding — the "
                         "per-finding representation-DEPTH axis (collaborator "
                         "cycle-7 idea 3; scratch/depthtap/). Probes read "
                         "GAP-pooled stage features through a stop-gradient, "
                         "train with their own AdamW, and save to a SEPARATE "
                         "fold{N}_stage_probes.pt — the main checkpoint stays "
                         "byte-compatible with every existing consumer. "
                         "Include the FINAL stage as the aggregation-matched "
                         "control: depth is judged probe-vs-probe, never "
                         "probe-vs-main-head.")
    ap.add_argument("--stage-tap-lr", type=float, default=1e-3)
    ap.add_argument("--save-subepoch", default="",
                    help="Comma-separated fractions in (0,1), e.g. '0.25,0.5,0.75'. "
                         "Saves an UNCONDITIONAL EMA snapshot that far through "
                         "the listed epochs. Empty (default) = off, and off "
                         "changes nothing. Rationale: the two things that have "
                         "ever moved LB by selection both say the same thing - "
                         "Synovitis is served by ep0-1 checkpoints, and the "
                         "fold-AUC@.5 selector beats softAUCn by +0.0026 "
                         "BECAUSE softn picks about one epoch late. Epoch "
                         "granularity is the floor of that axis; without "
                         "sub-epoch files the 'a5 minus half an epoch' bracket "
                         "cannot be read at all. These are scored offline, so "
                         "they carry no seed noise (post-hoc re-ranking over "
                         "fixed preds is +/-0.0013) and cost only disk.")
    ap.add_argument("--save-subepoch-epochs", default="0,1",
                    help="Which epochs --save-subepoch applies to.")
    # --- PF-RelationNet V0 (scratch/pfrel/PREREG.md, DESIGN_V0_20260917.md) ---
    # Mutually exclusive arms of ONE prereg. Both default OFF; off is the
    # pre-existing code path byte-for-byte.
    ap.add_argument("--pfrel", type=int, default=0, choices=[0, 1],
                    help="PFREL arm: the relational patella<->trochlea PF-OA "
                         "specialist (spec s2-s9). Replaces the slice/study "
                         "aggregation entirely; trains ONE column (PF OA).")
    ap.add_argument("--pfrel-generic", type=int, default=0, choices=[0, 1],
                    help="PF_GENERIC control arm (spec s11): identical "
                         "encoder/input/through-plane/head, six surface "
                         "queries replaced by one generic attention-pooled "
                         "128-d slice token, parameter-matched with a small "
                         "MLP. The primary statistic is Delta_REL = "
                         "AUC(PFREL) - AUC(PFGENERIC) on the honest PF column.")
    ap.add_argument("--pfrel-slices", type=int, default=24,
                    help="contiguous centre slices of the chosen axial "
                         "fluid-sensitive series (spec s1: 24-32).")
    ap.add_argument("--pfrel-lambda-pos", type=float, default=0.1,
                    help="init of the LEARNABLE positional-bias scale lambda "
                         "(spec s4: 5-10%% of the attention-logit scale). The "
                         "bias B(x,y) is fixed and weak; there is no ROI, mask "
                         "or segmenter output anywhere in this arm.")
    ap.add_argument("--pfrel-beta", type=float, default=1.0,
                    help="init of the centroid-distance penalty beta in the "
                         "relational soft matching (spec s5).")
    ap.add_argument("--pfrel-aux", type=float, default=0.02,
                    help="TOTAL weight of the two spec-s9 auxiliaries "
                         "(centroid smoothness + anti-collapse hinge), split "
                         "evenly. Spec allows 0.01-0.03x the classification "
                         "loss; 0.02 is the preregistered value.")
    ap.add_argument("--pfrel-init", default="",
                    help="checkpoint whose backbone.* weights initialise the "
                         "ConvNeXt trunk (spec s2: proven, fold-MATCHED "
                         "weights). Loaded backbone-only and non-strict by "
                         "design - nothing else in this model exists in a T1 "
                         "checkpoint - and the count of loaded tensors is "
                         "printed and asserted non-zero.")
    # --- Bone-WSRPN V0 (scratch/bonewsrpn/PREREG.md) ---
    # Mutually exclusive arms of ONE prereg. Both default OFF; off is the
    # pre-existing code path byte-for-byte.
    ap.add_argument("--bonewsrpn", type=int, default=0, choices=[0, 1],
                    help="BONE_ROI arm: K differentiable soft 3D lesion "
                         "proposals per series + core/shell pooling + smooth "
                         "MIL. Trains TWO columns (Contusion, Fracture).")
    ap.add_argument("--bone-dense", type=int, default=0, choices=[0, 1],
                    help="BONE_DENSE control arm: identical encoder, FPN, 3D "
                         "blocks, heads, factorisation, schedule and loss with "
                         "the proposals REMOVED (per-voxel logits + MIL over "
                         "voxels, parameter-matched). The primary statistic is "
                         "the pair-mean of (dCont + dFrac)/2, ROI - DENSE.")
    ap.add_argument("--bone-series-fluid", default="fs", choices=["fs", "nonfs"],
                    help="scratch/bonefam: DATA-path flag for the bone "
                         "specialist. fs (default) = sag FS + cor FS, "
                         "byte-identical to bonewsrpn; nonfs = sag + cor "
                         "NON-fluid-sensitive (T1/PD) series by the same "
                         "deterministic rule. Requires --bone-dense or "
                         "--bonewsrpn. State dict is identical, so every "
                         "consumer must read ck['args'] (refuse-not-skip).")
    ap.add_argument("--bone-k", type=int, default=4,
                    help="proposals per series (union = 2*K objects).")
    ap.add_argument("--bone-slices", type=int, default=24,
                    help="contiguous centre slices per series.")
    ap.add_argument("--bone-tau", type=float, default=0.1,
                    help="sharpness of the sigmoid-product soft 3D mask.")
    ap.add_argument("--ps25", action="store_true",
                    help="scratch/divarch: per-series 2.5D late-fusion model (mean+max slice pooling, per-series 12-logit head, masked-mean fusion over series). Not ported to the Kaggle kernel.")
    ap.add_argument("--bone-shell", type=float, default=2.0,
                    help="shell scale multiplier (shell = mask(shell*s) - "
                         "mask(s), clamped at 0).")
    ap.add_argument("--bone-aux", type=float, default=0.01,
                    help="weight of the proposal-diversity auxiliary (soft-"
                         "mask overlap), applied only where >= 2 objects have "
                         "objectness sigmoid > 0.5. The dense control has no "
                         "objects and adds nothing here.")
    ap.add_argument("--bone-init", default="",
                    help="checkpoint whose backbone.* weights initialise the "
                         "ConvNeXt trunk (fold-MATCHED). Loaded backbone-only "
                         "and non-strict; the loaded-tensor count is printed "
                         "and a zero count aborts the job.")
    ap.add_argument("--select-findings", default="",
                    help="comma-separated findings that checkpoint selection "
                         "scores on. For a specialist trunk (11 findings muted "
                         "from epoch 0) the all-column metric is 11/12 noise. "
                         "Empty = all twelve, i.e. every previous run's "
                         "behaviour. Example: --select-findings Synovitis")
    # --- grad-probe: does the ACTUAL AdamW step move the TRUNK against a task?
    # PREREG scratch/taskcompromise/PREREG.md. Instrumentation only: with
    # --grad-probe-epochs unset this costs one falsy check per epoch and cannot
    # touch the trajectory. Probe batches come from a SEPARATE deterministic
    # loader (mk(tr, False), shuffle=False), so training RNG is untouched and a
    # probed run is step-for-step identical to an unprobed one.
    ap.add_argument("--grad-probe-epochs", default="",
                    help="Comma list of epochs at which to measure the real "
                         "optimizer step against each finding's own gradient. "
                         "Writes g_j.dtheta and C_j = -(g_j.dtheta)/(|g_j||dtheta|) "
                         "for the SHARED TRUNK and the full model, plus the "
                         "raw -> preconditioned -> Adam-direction -> actual-step "
                         "ladder. Empty = OFF.")
    ap.add_argument("--grad-probe-out", default="",
                    help="Directory for grad-probe JSON (one file per epoch).")
    ap.add_argument("--grad-probe-batches", type=int, default=8,
                    help="Deterministic probe batches per checkpoint; the "
                         "DISTRIBUTION of C_j over these is what is reported, "
                         "not one noisy minibatch.")
    ap.add_argument("--mute-after", default="",
                    help="Comma list of FINDING:EPOCH[:WEIGHT]. From EPOCH on, "
                         "that finding's loss weight becomes WEIGHT (default 0). "
                         "Why: AUC(extracted grade -> gold) is an ASYMPTOTE, "
                         "because BCE converges to its targets - Synovitis's is "
                         "0.723 while the model reaches 0.84 at epoch 0 and "
                         "decays to 0.67. Calibration and loss weighting only "
                         "change how fast that decay happens; muting stops it, "
                         "and lets the shared trunk keep improving on the "
                         "eleven findings whose labels are worth fitting. "
                         "Example: --mute-after 'Synovitis:2'")
    ap.add_argument("--group-threshold", type=float, default=0.8,
                    help="Jaccard threshold for grouping near-duplicate reports "
                         "into the same fold. 1.0 reverts to exact-hash "
                         "grouping, which leaves ~10%% of each val fold with a "
                         "near-twin in train.")
    ap.add_argument("--laterality-map",
                    default="data/processed/laterality_map_v2.json",
                    help="v2 = DICOM tag + image classifier, 100%% coverage. "
                         "Recorded in the checkpoint so inference can detect a "
                         "mismatch: a model trained on v2 needs "
                         "laterality_clf.json shipped alongside it.")
    ap.add_argument("--agree-weights", default="",
                    help="npz of per-(study,finding) label confidence from "
                         "repeated extraction passes. Running the same prompt "
                         "three times disagrees on 10.4%% of cells - 26%% on "
                         "Fracture, 20%% on Contusion - and those cells are coin "
                         "flips. This downweights them PER CELL, which changes "
                         "the relative weight of studies within a finding; a "
                         "per-finding rescale would be a monotone transform and "
                         "could not change what the model ranks.")
    ap.add_argument("--agree-floor", type=float, default=0.3)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Trade compute for activation memory in the backbone. "
                         "convnextv2_tiny OOMs on a 44 GiB L40S at our contract "
                         "(6 series x 24 slices x 384px) despite having the same "
                         "27.9M params as convnext_tiny - its GRN layers hold "
                         "far more activation. Reducing n_slices or n_series "
                         "would fix it too but would change the INPUT CONTRACT, "
                         "making the weights unpoolable with the 5-fold - which "
                         "defeats the point, since this arm exists for ensemble "
                         "diversity.")
    ap.add_argument("--cotrain-pool", default="",
                    help="JSON {name: [uid, ...]} of EXTERNAL studies (e.g. "
                         "OAI) to train on. They join the TRAIN half after the "
                         "split and never enter `studies`, so the grouping, the "
                         "permutation and the val fold stay byte-identical to a "
                         "run without the flag - fold-AUC and OOF remain "
                         "comparable to every previous arm and to the 919-study "
                         "rig. Without this, ~1/n_folds of the pool would land "
                         "in VALIDATION, where --coverage is deliberately not "
                         "applied, so findings the external source never scored "
                         "(grade -1 -> 0.10) would be scored as fabricated soft "
                         "negatives on a different case mix.")
    ap.add_argument("--cotrain-epochs", default="",
                    help="LO:HI (float epochs) - pool studies contribute loss "
                         "ONLY while LO <= fractional_epoch < HI (B3 gradient-"
                         "provenance schedules: front-OAI '0:5', late-OAI "
                         "'5:10'; empty = constant, the incumbent behaviour). "
                         "Implemented as a loss-weight gate, NOT a dataset "
                         "change: the loader, steps-per-epoch, OneCycle "
                         "schedule, EMA cadence and sub-epoch snapshot points "
                         "stay byte-identical to the constant arm, so the ONLY "
                         "difference between B3 arms is WHEN pool gradients "
                         "flow. A gated-off sample still runs its forward "
                         "(zero-weight == excluded exactly: the stack is "
                         "LayerNorm-only, no batch statistics). Requires "
                         "--cotrain-pool.")
    ap.add_argument("--gold-train-labels", default="expert",
                    choices=["expert", "report"],
                    help="Which targets the gold-train rows carry. 'expert' = "
                         "incumbent behaviour (bit-identical default). "
                         "'report' = CONTROL arm of the gold-substitution "
                         "dose-response: identical rows and weight, report "
                         "targets kept, so T-C isolates label SOURCE at fixed "
                         "weighting/case-mix (scratch/goldsub/PREREG.md).")
    ap.add_argument("--gold-train-w", type=float, default=0.0,
                    help="Train on the 58 expert-labelled studies at this "
                         "sample weight (public notebooks use 3-4). They are "
                         "the only ground truth we have and are currently 100%% "
                         "unused for training. Requires --gold-holdout-folds: "
                         "the held-out slice is excluded from training AND is "
                         "the only slice GOLD-AUC is scored on, so the eval "
                         "stays honest. Run every fold and pool to recover a "
                         "full-58 estimate.")
    ap.add_argument("--gold-train-soft", action="store_true",
                    help="With --gold-train-labels expert, map the expert 0/1 "
                         "onto the prob_table ladder ends (.05/.95) instead of "
                         "hard 0/1. Default OFF = goldburn's convention, so "
                         "every pre-existing arm is byte-identical.")
    ap.add_argument("--gold-holdout-folds", type=int, default=5)
    ap.add_argument("--gold-holdout-fold", type=int, default=0)
    ap.add_argument("--no-gold", action="store_true",
                    help="Do not load Gold labels or construct/evaluate a Gold "
                         "dataset. Requires --study-allowlist-npz, whose uids "
                         "define the complete permitted internal study pool. "
                         "Gold-dependent calibration and Gold training are "
                         "hard-rejected.")
    ap.add_argument("--study-allowlist-npz", default="",
                    help="NPZ containing a unique `uids` vector. Under "
                         "--no-gold this replaces Gold-derived exclusion and "
                         "near-duplicate filtering; every UID must have labels "
                         "and volume metadata.")
    ap.add_argument("--rank-w", type=float, default=0.0,
                    help="NOT RECOMMENDED - measured to be a no-op that damages "
                         "the ordinal targets. For a fixed bank the gradient "
                         "depends only on the study's own logit and is "
                         "increasing in it, so the update is a MONOTONE map: it "
                         "compresses logits without ever reordering two "
                         "studies, and AUC is invariant to monotone maps. "
                         "Measured effect at 0.05: +0.00005. Worse, "
                         "PRIOR_TABLE has no value in [0.40,0.60], so the gate "
                         "does not exclude an ambiguous middle - it BINARISES, "
                         "forcing grade 1 (P(gold=1)=0.358) to the negative "
                         "side. The gate label scores 0.8282 on gold against "
                         "0.8772 for the ordinal target it replaces, Synovitis "
                         "-0.108. Kept only so the finding is reproducible. "
                         "Original rationale: the metric IS AUC, "
                         "but the training objective is BCE, which optimises "
                         "calibration rather than order - the mismatch we "
                         "already patched downstream with ceiling-normalised "
                         "soft-AUC selection. This attacks it at the source. "
                         "0.05 is what the public notebooks that are climbing "
                         "use. Needs pairs, and we train at bs=1, so it ranks "
                         "the current study against a FIFO of recent detached "
                         "logits (see --rank-bank).")
    ap.add_argument("--routing-prior", type=float, default=0.0,
                    help="additive attention-logit tilt toward each finding's "
                         "clinically-preferred series types (model.ROUTE_PRIOR). "
                         "0 = off, bit-identical to no prior. pilkwang v14 "
                         "ships the equivalent at 0.55.")
    ap.add_argument("--llrd", type=float, default=1.0,
                    help="layer-wise LR decay per backbone depth bucket "
                         "(top bucket = lr*backbone_lr_mult, each earlier "
                         "bucket multiplied by this). 1.0 = flat (current).")
    ap.add_argument("--plane-filter", default="",
                    choices=["", "Sagittal", "Coronal", "Axial"],
                    help="specialist member: restrict series selection to one "
                         "anatomical plane. Studies with no series in that "
                         "plane keep their full series list (fallback) so the "
                         "fold split and study set stay identical to control.")
    ap.add_argument("--exclude-manufacturer", default="",
                    help="leave-one-site-out: drop studies whose modal series "
                         "manufacturer contains this substring (case-"
                         "insensitive) from the TRAIN half only. The fold "
                         "split and val set are untouched, so per-site val "
                         "AUC on the held manufacturer measures unseen-site "
                         "generalisation against the standard fold control.")
    ap.add_argument("--rank-bank", type=int, default=256,
                    help="How many recent studies to rank against. Gradient "
                         "flows only through the current study's logits; the "
                         "bank is detached, so this costs a 256x12 buffer and "
                         "no extra backward.")
    ap.add_argument("--sam-rho", type=float, default=0.0,
                    help="Sharpness-aware minimisation ascent radius (0 = "
                         "off). Implemented per micro-batch (m=1 sharpness) "
                         "with a manual gradient accumulator so bs1/accum8/"
                         "AMP/EMA step indices stay identical to a plain "
                         "run. Doubles forward/backward cost. Plain-recipe "
                         "path only (soft_bce + aux + gold_train_w); every "
                         "other loss term is refused by assertion, not "
                         "silently dropped from the SAM gradient.")
    ap.add_argument("--stop-epoch", type=int, default=0,
                    help="Break out of training after this many epochs while "
                         "keeping the LR schedule built for --epochs, so the "
                         "early trajectory is schedule-matched to full runs "
                         "(0 = train all epochs). For trajectory kill-tests; "
                         "checkpoints saved as usual up to the break.")
    ap.add_argument("--domain-rand", type=float, default=0.0,
                    help="per-series probability of acquisition-style\n"
                         "randomization (Fourier warp, bias field, Rician\n"
                         "noise, k-space truncation, contrast warp, ghosting)")
    ap.add_argument("--pair-max", default="",
                    help="json {uid: grade} for an unsplit meniscus-tear label\n"
                         "(MRNet); supervises max(MedM, LatM) logits with BCE")
    ap.add_argument("--pair-max-w", type=float, default=0.2)
    ap.add_argument("--report-aux", default="",
                    help="jsonl from `extract_openrouter.py --spec aux`: eight "
                         "auxiliary report findings the competition does not "
                         "score, three of them compartment-localised cartilage. "
                         "Extra supervision for the shared trunk aimed at the "
                         "measured deficit - the model is near ceiling at "
                         "whether there is pathology (Medial OA 0.978) and weak "
                         "at where it is (Lateral OA 0.850, PF OA 0.835). "
                         "Training-only; test.csv has no Report column.")
    ap.add_argument("--report-aux-w", type=float, default=0.2)
    ap.add_argument("--moaks-aux", default="",
                    help="jsonl from src/build_moaks_subregion_targets.py: the "
                         "62-column per-subregion MOAKS target vector "
                         "(cartilage area+depth, osteophyte, BML, extrusion, "
                         "compartment rubric summaries) as masked auxiliary "
                         "supervision on the OAI co-training rows "
                         "(scratch/latoarescue/PREREG.md). One linear head per "
                         "aux block off the study representation; grades map "
                         "through the SAME production ladder as every other "
                         "source. Default OFF and byte-path-inert when absent. "
                         "Mutually exclusive with --report-aux. Training-only; "
                         "heads are dropped at serve time.")
    ap.add_argument("--moaks-aux-w", type=float, default=0.2)
    ap.add_argument("--moaks-aux-holdout", type=int, default=10,
                    help="1-in-N KNEE-level holdout (md5 of subject-side, all "
                         "visits of a knee together) excluded from MOAKS aux "
                         "supervision and scored at each epoch end for the "
                         "learnability gate (PREREG 4a). The held-out rows "
                         "keep their competition-label training role, so the "
                         "training uid set is untouched. 0 disables.")
    ap.add_argument("--soft-targets", default="",
                    help="npz from src/build_self_training_labels.py: blends "
                         "the model's out-of-fold predictions into the report "
                         "targets, per finding, weighted by how far the model "
                         "beats that finding's label ceiling. Justified only "
                         "where it does - ACL and MCL sit BELOW their ceilings, "
                         "so their alpha is 0 and their labels are untouched.")
    ap.add_argument("--teacher-targets", default="",
                    help="NPZ of pure image-teacher probabilities (uids, "
                         "targets, findings). Used by --teacher-curriculum "
                         "as a LIVE training-step target; report labels and "
                         "validation labels are never overwritten. The file "
                         "must contain one finite (N,12) target per unique UID "
                         "and must exclude all 58 gold studies.")
    ap.add_argument("--teacher-curriculum", default="off",
                    choices=["off", "fixed", "late"],
                    help="live report/teacher target schedule: off (default), "
                         "fixed (teacher weight --teacher-max-w at every "
                         "step), or late (0 before epoch 2, linear ramp to "
                         "--teacher-max-w at epoch 5, then held). The late "
                         "schedule is the preregistered C arm; fixed is B.")
    ap.add_argument("--teacher-max-w", type=float, default=0.5,
                    help="maximum teacher target weight for --teacher-curriculum "
                         "(preregistered B/C value: 0.5).")
    ap.add_argument("--teacher-ramp-start", type=float, default=2.0,
                    help="fractional epoch at which the late teacher ramp "
                         "starts (preregistered: 2).")
    ap.add_argument("--teacher-ramp-end", type=float, default=5.0,
                    help="fractional epoch at which the late teacher ramp "
                         "reaches --teacher-max-w (preregistered: 5).")
    ap.add_argument("--distill-logit-mse", type=float, default=0.0,
                    help="W: ADD W*MSE(z_student, logit(clip(p_teacher, 1e-4, "
                         "1-1e-4))) to the existing soft-BCE, on exactly the "
                         "TRAIN rows that --soft-targets actually replaced. "
                         "soft-BCE's gradient is sigma(z)-p, which VANISHES "
                         "where the teacher is confident; the competition "
                         "metric is pure ranking, so the ordering inside those "
                         "saturated tails is precisely what AUC rewards and "
                         "precisely what BCE under-fits at finite training "
                         "time. Logit MSE has uniform gradient across the "
                         "range. 0 = off, and at 0 the loss line is "
                         "byte-identical to every pre-existing arm's. "
                         "Prereg: scratch/effwave/PREREG.md (arm effL, W=0.5).")
    ap.add_argument("--aux-seg-w", type=float, default=0.0,
                    help="W: ADD W*(CE+soft-Dice) auxiliary segmentation loss "
                         "from a light decoder off the encoder's last feature "
                         "map, supervised by 6-class anatomical pseudo-masks "
                         "under --aux-seg-root (scratch/anatroi/masks_all). "
                         "0.0 = off: no decoder is built (model state_dict "
                         "unchanged), the dataset never opens a mask file, "
                         "and the classifier loss line is byte-identical to "
                         "every pre-existing arm's. Discarded at serve time. "
                         "Prereg: scratch/auxdice/PREREG.md.")
    ap.add_argument("--slice3d", default="none",
                    choices=["none", "shift", "inflate"],
                    help="through-plane feature mixing INSIDE the ConvNeXt "
                         "encoder (scratch/slice3d/PREREG.md): every block's "
                         "depthwise conv is wrapped so the slice axis is "
                         "recovered from the flattened batch and mixed with "
                         "zero-padded neighbours. shift = zero-parameter "
                         "temporal shift of C/8+C/8 channels; inflate = I3D "
                         "centre-init 3x7x7 depthwise (two new zero-init "
                         "taps per block). none (default) installs no "
                         "wrapper: module graph and state_dict identical to "
                         "every pre-existing arm. Saved in ck['args']; "
                         "consumers must rebuild with it.")
    ap.add_argument("--blurpool", type=int, default=0, choices=[0, 1],
                    help="scratch/regbundle/PREREG.md: 1 = anti-aliased "
                         "downsampling (fixed binomial 3x3 depthwise blur "
                         "before the ConvNeXt stem conv and each stage "
                         "downsample conv; zero params, state_dict "
                         "unchanged). 0 (default) installs nothing. Saved in "
                         "ck['args']; consumers must rebuild with it.")
    ap.add_argument("--qsa", default="none", choices=["none", "pre", "post"],
                    help="inter-finding query self-attention "
                         "(scratch/qsa/PREREG.md): ONE pre-norm MHA block over "
                         "the 12 scored queries, residual through a learned "
                         "scalar gate (--qsa-gate-init). post = on the routed "
                         "per-finding vectors after cross-attention, before "
                         "the heads (study-dependent exchange; the mechanism "
                         "arm). pre = on the query parameters before "
                         "cross-attention (static; equal-parameter capacity "
                         "placebo). none (default) builds no module: "
                         "state_dict and forward identical to every "
                         "pre-existing arm. Saved in ck['args']; consumers "
                         "must rebuild with it.")
    ap.add_argument("--qsa-gate-init", type=float, default=0.1,
                    help="initial value of the qsa residual gate; must be "
                         "non-zero (zero-init gates collapse here).")
    ap.add_argument("--lr-tail-frac", type=float, default=0.0,
                    help="scratch/consttail/PREREG.md (C2, constant-LR tail "
                         "with EMA as the only annealer): fraction F of the "
                         "OneCycle schedule over which the LR is HELD at the "
                         "value reached at step round((1-F)*steps) instead of "
                         "continuing the cosine anneal to ~0. Loader, steps, "
                         "EMA and everything else are untouched; EMA keeps "
                         "averaging over the held-LR tail. 0 (default) = the "
                         "incumbent full anneal, bit-identical (the hold step "
                         "is then `steps`, subsumed by the existing "
                         "`last_epoch < steps-1` guard). Saved in ck['args'].")
    ap.add_argument("--drop-path", type=float, default=0.0,
                    help="scratch/consttail/PREREG.md (C3, stochastic depth): "
                         "timm drop_path_rate for the backbone blocks. 0 "
                         "(default) passes NOTHING to create_model = the "
                         "incumbent (timm default off), bit-identical. "
                         "Identity at eval; state_dict unchanged, so consumers "
                         "need not rebuild with it. Saved in ck['args'].")
    ap.add_argument("--mmix-p", type=float, default=0.0,
                    help="scratch/regbundle/PREREG.md: per-step probability "
                         "of manifold mixup at the last hidden layer (12 "
                         "study-query vectors) against a detached bank of the "
                         "previous training study (bs=1 cross-step form). "
                         "Targets/weights mixed with the same per-cell lam on "
                         "cells supervised in BOTH studies. Train-only; 0 "
                         "(default) executes no extra op.")
    ap.add_argument("--mmix-alpha", type=float, default=0.4,
                    help="Beta(alpha, alpha) for --mmix-p.")
    ap.add_argument("--dismix-p", type=float, default=0.0,
                    help="scratch/dismix/PREREG.md (src/dismix.py): per-CELL "
                         "probability of ordinal-adjacent mixup on the 12 "
                         "pre-head query vectors -- finding k mixes ONLY with "
                         "a banked study whose grade on k is one ladder step "
                         "away; targets/weights mixed with the same lam; the "
                         "other rows untouched. Train-only; 0 (default) "
                         "executes no extra op (byte-identical path).")
    ap.add_argument("--dismix-alpha", type=float, default=1.0,
                    help="Beta(alpha, alpha) for --dismix-p (1.0 = uniform).")
    ap.add_argument("--dismix-max-age", type=int, default=256,
                    help="oldest bank entry (in training studies) --dismix-p "
                         "may pair with; older adjacent partners are skipped.")
    # --- forktest (scratch/forktest/PREREG.md): target-only continuation forks.
    # All three flags are DEFAULT-INERT and architecture-inert: they change the
    # schedule, the stopping point and the per-column loss weights only, never
    # model construction, so src/ckpt_build.py needs no new key (the equivalence
    # gate, --max-steps 0, proves the built model is the snapshot's).
    ap.add_argument("--const-lr", type=float, default=0.0,
                    help="forktest: >0 replaces the OneCycle schedule with a "
                         "CONSTANT LR. Each param group is set to "
                         "const_lr * (group max_lr / max(max_lr)), so the "
                         "backbone/head ratio (--backbone-lr-mult, --llrd) is "
                         "preserved; sched.step() is then never called.")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="forktest: >=0 stops training after this many "
                         "OPTIMIZER steps (0 = no training at all, which is "
                         "the equivalence gate: with --resume-model the run "
                         "must reproduce the snapshot's predictions exactly). "
                         "The epoch-end validation/save still runs, so a fork "
                         "writes exactly one fold{N}_best.pt / _ema.pt / "
                         "_oof.npz with no cross-epoch selection. -1 = off.")
    ap.add_argument("--loss-cols", default="",
                    help="forktest: comma list of FINDINGS that keep loss "
                         "weight; every other column is weighted 0 from epoch "
                         "0. Implemented through the existing --mute-after "
                         "machinery, i.e. the mask is applied at the LOSS "
                         "(per-column weight lw), never at the logits, so a "
                         "factorised column (--osseous-factor couples "
                         "Contusion/Fracture through shared oedema logits) "
                         "still receives gradient through its own path. Empty "
                         "(default) = all 12, byte-identical. Mutually "
                         "exclusive with --mute-after.")
    # --- recipe2 (scratch/recipe2/PREREG.md): two flag-gated recipe arms.
    ap.add_argument("--opt", default="adamw", choices=["adamw", "ademamix"],
                    help="adamw (default, byte-identical path) or ademamix "
                         "(src/opt_ademamix.py: AdamW + slow gradient EMA "
                         "mixed into the numerator, ICLR 2025).")
    ap.add_argument("--ademamix-alpha", type=float, default=5.0)
    ap.add_argument("--ademamix-beta3", type=float, default=0.999)
    ap.add_argument("--ademamix-warmup", type=float, default=1.0,
                    help="alpha/beta3 warmup length as a FRACTION of total "
                         "optimizer steps (paper: T_alpha = T_beta3 = T).")
    ap.add_argument("--pcol-w", type=float, default=0.0,
                    help="src/loss_pcol.py: weight of the auxiliary prototype "
                         "contrastive ordinal loss on the 12 pre-head query "
                         "embeddings (graded, non-silent, weighted cells). 0 "
                         "(default) constructs nothing and adds no op.")
    ap.add_argument("--pcol-tau", type=float, default=0.1)
    ap.add_argument("--pcol-sigma", type=float, default=1.0)
    ap.add_argument("--aux-seg-root", default="",
                    help="root dir of <StudyUID>/<SeriesUID>_msk.npy 6-class "
                         "masks for --aux-seg-w (e.g. "
                         "scratch/anatroi/masks_all). Required when "
                         "--aux-seg-w > 0.")
    ap.add_argument("--series-dropout", type=float, default=0.0,
                    help="p of dropping one present series per study during training\n                    (protocol-composition robustness); 0 = off.")
    ap.add_argument("--affine-p", type=float, default=0.7,
                    help="probability of the base affine augmentation "
                         "(rotation/scale/shift). 0 = off.")
    ap.add_argument("--spatial-aug", type=float, default=0.0,
                    help="p of zoom-in crop (area 0.8-1.0, per series) + "
                         "per-slice random erasing. The two spatial ops the "
                         "default affine does not cover; 0 = off.")
    ap.add_argument("--vendor-aug", type=float, default=0.0,
                    help="p of vendor-spanning intensity augmentation per series:\n                    gamma/gain/offset/noise. Site-shift hardening; 0 = off.")
    ap.add_argument("--cutmix", type=float, default=0.0,
                    help="Probability of CutMix-ing a training study with a "
                         "random partner. RSNA 2025's 5th place measured cutmix "
                         "p=1.0 as the best augmentation FOR AUC specifically - "
                         "beating flip, rotate, brightness, contrast, blur and "
                         "mixup - and their 6th place put cutmix/mixup at about "
                         "+0.01 CV. Our policy before this was rotate +/-10, "
                         "scale 0.92-1.08, shift +/-4%% at p=0.7, which is mild "
                         "by their standards. Mixed in the DATASET because bs=1 "
                         "leaves no partner inside a batch.")
    ap.add_argument("--cutmix-alpha", type=float, default=1.0,
                    help="Beta(a,a) shape for the CutMix area. 1.0 is uniform.")
    ap.add_argument("--band-pool", action="store_true",
                    help="Replace the backbone's global spatial pool with "
                         "left/centre/right bands along the image column axis, "
                         "concatenated with the global vector. On coronal and "
                         "axial that axis IS medial vs lateral, which the global "
                         "pool currently averages away - our largest measured "
                         "gap. RSNA 2025's 6th place used exactly this for 13 "
                         "anatomical locations with NO localisation labels. The "
                         "projection is zero-initialised on the bands and "
                         "identity on the global vector, so step 0 is "
                         "bit-identical to the model without it.")
    ap.add_argument("--flip-swap", type=float, default=0.0,
                    help="Probability of mirroring a training study AND swapping "
                         "its paired labels (Medial<->Lateral Meniscus, "
                         "Medial<->Lateral OA). A plain horizontal flip was "
                         "removed from this codebase because it teaches the "
                         "wrong map; a flip WITH the swap is a different "
                         "operation and is what every top team with a left/right "
                         "target pair used - RSNA 2025 5th measured +0.01, RSNA "
                         "2024 3rd said it 'effectively doubled the amount of "
                         "data'. Targets our prevalence asymmetry directly: "
                         "medial meniscus is 56.8%% of the corpus against "
                         "lateral's 27.2%%, and after the swap both queries see "
                         "the same volume. MCL and Baker's are laterality-"
                         "specific with no counterpart, so their loss is masked "
                         "on flipped studies. Depends on the laterality map "
                         "being right - a wrong side silently swaps medial and "
                         "lateral.")
    ap.add_argument("--slice-ctx", type=int, default=3,
                    help="How many adjacent slices are stacked as channels. "
                         "Fixed at 3 since the first run. RSNA 2023's 6th place "
                         "reported that 1 slice with +/-2 as FIVE channels beat "
                         "their standard multi-slice construction, and were "
                         "surprised by it. This also widens the only positional "
                         "information the model has: the slice transformer has "
                         "no positional encoding, so the +/-k neighbourhood is "
                         "its entire sense of through-plane order. NOTE this "
                         "changes the backbone's in_chans, so the weights are "
                         "NOT interchangeable with slice_ctx=3 checkpoints.")
    ap.add_argument("--bypass-study-transformer", action="store_true",
                    help="scratch/evmil_bypass: attribution control for "
                         "--evidence-mil. Keeps the slice attention pool, the "
                         "series/type embeddings, the 12 cross-attention "
                         "queries, the heads and the train-only aux head, and "
                         "removes ONLY the inter-series study-transformer "
                         "self-attention/FFN layers (series tokens reach the "
                         "queries unchanged). Saved in ck['args']; consumers "
                         "rebuild through src/ckpt_build.py.")
    ap.add_argument("--study-pool", default="route", choices=["route", "mean"],
                    help="scratch/stackdepth arm E: 'mean' replaces the study "
                         "transformer + 12-query routing with ONE masked mean "
                         "over series shared by all findings. Deletes the "
                         "anatomical routing prior and the family embedding, "
                         "both measured parts of the shipped edge, so expect it "
                         "to cost routing-dependent columns. Parameter-shape "
                         "compatible but behaviourally different: every "
                         "consumer must read it from ck['args'].")
    ap.add_argument("--stack-depth", action="store_true", default=False,
                    help="scratch/stackdepth: feed the n_slices depth axis as "
                         "the backbone's CHANNEL axis (one forward per SERIES) "
                         "instead of stacking slice_ctx neighbours per slice "
                         "(one forward per slice). Through-plane mixing then "
                         "happens inside the convolutions from layer one; the "
                         "slice transformer sees a length-1 sequence and is "
                         "inert, and the study transformer is unchanged. "
                         "REQUIRES --slice-ctx == --n-slices. Parameter-free, "
                         "so the state dict cannot reveal it: every consumer "
                         "must read it from ck['args'].")
    ap.add_argument("--stratified-slices", action="store_true", default=False,
                    help="Training only: sample one centre independently from "
                         "each depth bin. Evaluation retains the deterministic "
                         "grid; use matched fixed native context for comparisons.")
    ap.add_argument("--evidence-sampling-map", default="",
                    help="verified native-to-cache support intervals for the "
                         "conditional evidence-sampling arm; training only, "
                         "and inference sampling remains unchanged")
    ap.add_argument("--ctx-mm", type=float, default=0.0,
                    help="scratch/physgeo: with --native-ctx, place the context "
                         "channels +-ctx_mm mm from the centre slice (rounded to "
                         "whole native slices, min 1) using per-series thickness. "
                         "0 = off (byte-identical).")
    ap.add_argument("--aspect-fix", action="store_true", default=False,
                    help="scratch/physgeo: undo the anisotropic stretch of the "
                         "384-square resize for non-square acquisition matrices.")
    ap.add_argument("--xfuse", action="store_true", default=False,
                    help="scratch/xfuse: same-voxel FS/non-FS early fusion. "
                         "Each fat-suppressed series with a geometry-matched "
                         "non-FS partner (--xfuse-pairs table) is fed as a "
                         "2*slice_ctx-channel stem input on the same slice "
                         "grid; unpaired series zero-fill the partner half. "
                         "DATA-path + stem flag: stored in ck['args'], built "
                         "by src/ckpt_build.py, REFUSED by the kernel.")
    ap.add_argument("--xfuse-pairs", default="scratch/xfuse/xfuse_pairs.json",
                    help="scratch/xfuse/build_pairs.py output (only read with --xfuse).")
    ap.add_argument("--full-fov", action="store_true", default=False,
                    help="scratch/ffov: for size < 384, RESIZE the whole 384 "
                         "frame to size x size (INTER_AREA) instead of cropping "
                         "it (random/centre). Off = byte-identical loader. "
                         "Stored in ck['args']; every consumer (export/eval/"
                         "serve notebook) must read it from there.")
    ap.add_argument("--tfcrop-root", default="",
                    help="scratch/tfcrop_20261011: dir of <study>.npz native-resolution "
                         "lateral/medial TF coronal crops, APPENDED as 4 extra series "
                         "(type ids 6-9) after the n_series standard slots. Requires "
                         "--evidence-mil. Empty (default) = off, byte-inert.")
    ap.add_argument("--native-ctx", action="store_true", default=False,
                    help="scratch/nativectx: the loader returns (n_slices, "
                         "slice_ctx, H, W) windows whose channels are the centre "
                         "slice's NATIVE neighbours in the cached volume (clipped), "
                         "instead of KneeNet._stack_ctx pairing neighbours of the "
                         "linspace 24-stack. Series with z <= n_slices are "
                         "bit-identical either way. Stored in ck['args']; every "
                         "consumer (export/eval/serve) must read it from there - "
                         "the state dict cannot reveal it.")
    ap.add_argument("--slice-pos", default="none",
                    choices=["none", "shared", "type"],
                    help="Give the slice transformer a positional embedding. "
                         "Without it SeriesEncoder is EXACTLY permutation-"
                         "invariant over slices (measured 1.2e-07), and the "
                         "backbone already pools H and W away, so the model has "
                         "no representation of WHERE anything is - only the "
                         "+/-1 neighbourhood from the 3-slice channel stack. "
                         "Slice index is anatomy: after canonicalisation it "
                         "runs medial->lateral on sagittal, ant->post on "
                         "coronal, sup->inf on axial. 'type' gives each series "
                         "type its own table, since those are different axes; "
                         "'shared' makes all six share one. Targets the "
                         "compartment gap (medial findings beat their lateral "
                         "twins by 0.12-0.13 in 4/4 folds) without the masks or "
                         "the ~100 GPU-hours a segmentation head would cost - "
                         "but note that gap measured +0.007 on OOF, not the "
                         "+0.024 the gold-58 view suggested, so size it there.")
    ap.add_argument("--use-sequence", action="store_true",
                    help="Add a pulse-sequence embedding (PD/T1/T2/STIR/unknown) "
                         "parsed from SeriesDescription, alongside the existing "
                         "(plane x fluid-sensitive) type. Those 6 types conflate "
                         "sequences that look nothing alike - a T1 and a PD "
                         "coronal without fat-sat are the same token today, yet "
                         "T1 shows marrow and PD shows menisci and cartilage. "
                         "Resolves 80.7%% of series (98.1%% of those carrying a "
                         "description). Changes the input contract, so these "
                         "weights cannot be pooled with older ones; the "
                         "checkpoint records it and inference reads it back.")
    ap.add_argument("--use-geometry", action="store_true",
                    help="resample each series to a common mm-per-pixel. OFF by "
                         "default because every checkpoint so far was trained "
                         "without it; the flag is recorded in the checkpoint so "
                         "inference can match. Turning it on changes the input "
                         "contract - do not mix with older weights.")
    ap.add_argument("--keep-gold-dupes", action="store_true",
                    help="keep training studies whose report is byte-identical "
                         "to a gold study's (default drops them)")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="scratch/dinofam dinolora arm: >0 wraps every attention "
                         "q/v projection of the backbone with rank-r LoRA "
                         "(src/lora.py) and requires --freeze-backbone. 0 = off, "
                         "byte-identical path. Checkpoints keep lora_* keys; "
                         "consumers read `python src/lora.py merge`d files.")
    ap.add_argument("--lora-alpha", type=float, default=32.0,
                    help="LoRA scale numerator; effective scale = alpha/rank")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="requires_grad=False on every backbone.* parameter except "
                         "LoRA adapters; frozen tensors are excluded from the "
                         "optimizer by the existing requires_grad filter.")
    ap.add_argument("--freeze-bn", action="store_true",
                    help="scratch/resnetlr: after every model.train() put every "
                         "nn.BatchNorm* module inside model.backbone back into "
                         "eval() so the PRETRAINED running statistics are used "
                         "(never updated by <=6-image micro-batches at bs1); the "
                         "affine weight/bias still train. Default off = byte-"
                         "identical. No-op on BN-free trunks (ConvNeXt).")
    ap.add_argument("--seed", type=int, default=-1,
                    help="opt-in reproducible RNG seed; -1 preserves legacy behavior")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="seed of the GROUP PERMUTATION that deals the k folds "
                         "(default 42 = every checkpoint ever trained here). "
                         "--seed does NOT touch this: it seeds init/aug through "
                         "the global RNGs, while the deal uses its own "
                         "RandomState. Changing it RE-DEALS the split: same "
                         "study pool, same grouping, a different partition, so "
                         "fold-AUC is NOT comparable to a default-seed run and "
                         "OOF from the two deals must never be pooled. "
                         "Consumers (export_oof, flipswap_tta, tta_site_adapt, "
                         "build_nested_targets) read it back off ck['args'].")
    args = ap.parse_args()
    if args.tfcrop_root:
        if not os.path.isdir(args.tfcrop_root):
            raise SystemExit(f"--tfcrop-root {args.tfcrop_root} is not a directory")
        KneeStudyDataset.TFCROP_ROOT = args.tfcrop_root
        print(f"[tfcrop] ON root={args.tfcrop_root} extra series slots=4", flush=True)
    _xf_pairs = None
    if args.xfuse:
        from data import load_xfuse_pairs
        _xf_pairs = load_xfuse_pairs(args.xfuse_pairs)
        print(f"[xfuse] {len(_xf_pairs)} paired FS series from {args.xfuse_pairs}",
              flush=True)

    # --- PF-RelationNet V0 gate (scratch/pfrel/PREREG.md) --------------------
    # One mode string derived from two mutually exclusive flags, so the model,
    # the loader and ck["args"] can never disagree about which arm ran.
    if args.pfrel and args.pfrel_generic:
        raise SystemExit("--pfrel and --pfrel-generic are the two ARMS of one "
                         "prereg and cannot both be on")
    _pfrel_mode = "rel" if args.pfrel else ("generic" if args.pfrel_generic
                                            else "off")
    if _pfrel_mode != "off":
        # Positional verification of the single served column. 4-char finding
        # abbreviations collide (CLAUDE.md cluster tripwire); never parse by
        # name prefix.
        import model as _model_mod
        if FINDINGS[_model_mod.PF_OA_IDX] != "PF OA":
            raise SystemExit(
                f"model.PF_OA_IDX={_model_mod.PF_OA_IDX} is "
                f"{FINDINGS[_model_mod.PF_OA_IDX]!r}, not 'PF OA'")
        # The input contract is part of the arm, not a launcher choice: ONE
        # axial fluid-sensitive series, native +/-1 context, 384. Refuse rather
        # than silently coerce - a coerced flag would make the [args] line lie.
        if args.n_series != 1:
            raise SystemExit("--pfrel requires --n-series 1 (spec s1)")
        if not args.native_ctx:
            raise SystemExit("--pfrel requires --native-ctx (spec s1)")
        if args.n_slices != args.pfrel_slices:
            raise SystemExit(f"--pfrel: --n-slices {args.n_slices} != "
                             f"--pfrel-slices {args.pfrel_slices}")
        if not (24 <= args.pfrel_slices <= 32):
            raise SystemExit("--pfrel-slices must be 24-32 (spec s1)")
        if args.aux_w != 0.0:
            raise SystemExit("--pfrel has no per-series aux head (S == 1); "
                             "launch with --aux-w 0")
        # Only PF OA is trained. Muting is done through the EXISTING
        # --mute-after machinery (never through a label-file swap), and the
        # launcher must say so explicitly so the [args] line records it.
        _muted = {sp.split(":")[0].strip()
                  for sp in args.mute_after.split(",") if sp.strip()}
        _want = set(FINDINGS) - {"PF OA"}
        if _want - _muted:
            raise SystemExit(
                "--pfrel trains ONE column; mute the other eleven from epoch 0 "
                "with --mute-after: missing " + ", ".join(sorted(_want - _muted)))
        if args.select_findings.strip() != "PF OA":
            raise SystemExit("--pfrel requires --select-findings 'PF OA' "
                             "(the other eleven columns are constant zero)")

    # --- Bone-WSRPN V0 gate (scratch/bonewsrpn/PREREG.md) -------------------
    if args.bonewsrpn and args.bone_dense:
        raise SystemExit("--bonewsrpn and --bone-dense are the two ARMS of one "
                         "prereg and cannot both be on")
    _bone_mode = "roi" if args.bonewsrpn else ("dense" if args.bone_dense
                                               else "off")
    if args.bone_series_fluid != "fs" and _bone_mode == "off":
        raise SystemExit("--bone-series-fluid nonfs needs --bone-dense or "
                         "--bonewsrpn (scratch/bonefam)")
    if _bone_mode != "off":
        if _pfrel_mode != "off":
            raise SystemExit("--bonewsrpn and --pfrel are different specs")
        import model as _model_mod
        # Positional verification of the two served columns; 4-char finding
        # abbreviations collide (CLAUDE.md cluster tripwire).
        if (FINDINGS[_model_mod.CONT_IDX], FINDINGS[_model_mod.FRAC_IDX]) \
                != ("Contusion", "Fracture"):
            raise SystemExit(
                f"model.CONT_IDX/FRAC_IDX are "
                f"{FINDINGS[_model_mod.CONT_IDX]!r}/"
                f"{FINDINGS[_model_mod.FRAC_IDX]!r}, not Contusion/Fracture")
        if args.n_series != 2:
            raise SystemExit("--bonewsrpn requires --n-series 2 (sag FS + cor FS)")
        if not args.native_ctx:
            raise SystemExit("--bonewsrpn requires --native-ctx")
        if args.n_slices != args.bone_slices:
            raise SystemExit(f"--bonewsrpn: --n-slices {args.n_slices} != "
                             f"--bone-slices {args.bone_slices}")
        if args.aux_w != 0.0:
            raise SystemExit("--bonewsrpn has no per-series aux head; "
                             "launch with --aux-w 0")
        if args.osseous_factor:
            raise SystemExit("--bonewsrpn applies the osseous factorisation "
                             "itself, in BOTH arms; --osseous-factor would "
                             "apply it twice")
        _muted = {sp.split(":")[0].strip()
                  for sp in args.mute_after.split(",") if sp.strip()}
        _want = set(FINDINGS) - {"Contusion", "Fracture"}
        if _want - _muted:
            raise SystemExit(
                "--bonewsrpn trains TWO columns; mute the other ten from "
                "epoch 0 with --mute-after: missing "
                + ", ".join(sorted(_want - _muted)))
        if {t.strip() for t in args.select_findings.split(",") if t.strip()} \
                != {"Contusion", "Fracture"}:
            raise SystemExit("--bonewsrpn requires "
                             "--select-findings 'Contusion,Fracture'")

    # --- the DERIVED mode strings must reach ck["args"] ---------------------
    # The checkpoint stores vars(args), and src/ckpt_build.py reads the mode
    # under the keys "pfrel" and "bone" (refuse-not-skip: a checkpoint whose
    # weights carry pfrel_net.*/bone_net.* while args says "off" is REFUSED).
    # _pfrel_mode/_bone_mode were locals only, so every specialist checkpoint
    # saved args.pfrel=1/args.bonewsrpn=1 and NO mode key -> resolve_flags fell
    # back to "off" and the reload died ("bone mismatch ... says 'off' but the
    # weights carry bone_net.*", bone smoke 60155398). Nothing downstream reads
    # args.pfrel after this point (the int flag is consumed by the gate above),
    # so overwriting it with the mode string is safe and keeps ONE contract.
    args.pfrel = _pfrel_mode
    args.bone = _bone_mode

    device = "cuda"
    # auxvis: (6 series types, 12 findings) visibility table on device, or
    # None (production). _vis_acc = [kept cells, total present cells] per epoch.
    _vis_tab = None
    _vis_acc = [0.0, 0.0]
    if args.aux_visibility_mask:
        from aux_visibility import visibility_table, describe
        _vis_tab = torch.from_numpy(visibility_table()).to(device)
        print(describe(), flush=True)

    def _aux_vis_loss(aux_, y_, m_, wa_, stype_):
        """Visibility-masked per-series aux BCE: cells where the finding is
        not assessable on the series type get zero loss and are dropped from
        the denominator (per-cell mean over kept cells)."""
        v = m_ * _vis_tab[stype_]                       # (B,S,12)
        _vis_acc[0] += float(v.sum())
        _vis_acc[1] += float(m_.sum()) * y_.shape[1]
        return (torch.nn.functional.binary_cross_entropy_with_logits(
            aux_, y_.unsqueeze(1).expand_as(aux_), reduction="none")
            * v * wa_).sum() / v.sum().clamp(min=1)

    # --- evdrop (scratch/evdrop/PREREG.md) --------------------------------
    # Refuse-not-skip. The masked-view loss is added at exactly ONE site (the
    # standard per-microstep study loss), so every path that computes the
    # study loss somewhere else - the bagauc grad-cache local loss, the SAM
    # second pass, the label-repair interval loss - and every path that
    # rewrites `logits` after the forward (mmix, dismix) is refused rather
    # than silently half-applied.
    if args.evdrop:
        if not args.evidence_mil:
            raise SystemExit("--evdrop needs --evidence-mil: the masking acts "
                             "on the per-slice evidence tensor")
        _ed_bad = [n for n, v in (("--bagauc-w", args.bagauc_w > 0),
                                  ("--sam-rho", args.sam_rho > 0),
                                  ("--mmix-p", args.mmix_p > 0),
                                  ("--dismix-p", args.dismix_p > 0),
                                  ("--label-repair-sidecar",
                                   bool(args.label_repair_sidecar)),
                                  ("--evidence-mil-v2",
                                   str(args.evidence_mil_v2) != "off"),
                                  ("--syno-specialist",
                                   bool(args.syno_specialist))) if v]
        if _ed_bad:
            raise SystemExit(
                f"--evdrop is not isolated from {_ed_bad}; the masked-view "
                f"loss is added at one site only and these change where or "
                f"in what space the study loss is computed")
        if not (0.0 < args.evdrop_p <= 1.0):
            raise SystemExit(f"--evdrop-p must be in (0, 1], got "
                             f"{args.evdrop_p!r}")
    _evd_ct = [0]
    # --- PAT (scratch/shortcutarms/PREREG.md §PAT) -------------------------
    # Refuse-not-skip, the evdrop list plus every other train-only arm that
    # reads out[1] or hooks ev_head (the patch view calls ev_head a second
    # time, which a slicewit tap would capture).
    if args.pat:
        if not args.evidence_mil:
            raise SystemExit("--pat needs --evidence-mil")
        _pat_bad = [n for n, v in (("--bagauc-w", args.bagauc_w > 0),
                                   ("--sam-rho", args.sam_rho > 0),
                                   ("--mmix-p", args.mmix_p > 0),
                                   ("--dismix-p", args.dismix_p > 0),
                                   ("--label-repair-sidecar",
                                    bool(args.label_repair_sidecar)),
                                   ("--evidence-mil-v2",
                                    str(args.evidence_mil_v2) != "off"),
                                   ("--syno-specialist",
                                    bool(args.syno_specialist)),
                                   ("--evdrop", bool(args.evdrop)),
                                   ("--ev-adapter",
                                    str(args.ev_adapter) != "off"),
                                   ("--witness-targets",
                                    bool(args.witness_targets)),
                                   ("--slicewit-targets",
                                    bool(args.slicewit_targets))) if v]
        if _pat_bad:
            raise SystemExit(f"--pat is not isolated from {_pat_bad}")
        if args.pat_grid < 2 or args.n_slices % args.pat_grid:
            raise SystemExit(f"--pat-grid {args.pat_grid} must be >= 2 and "
                             f"divide --n-slices {args.n_slices}")
        if not (args.pat_w > 0):
            raise SystemExit(f"--pat-w must be > 0, got {args.pat_w!r}")
    _pat_ct = [0]
    # --cf-consist validation + resolved (finding index, masked half) pairs.
    _CF_ALIAS = {"LatOA": "Lateral OA", "MedOA": "Medial OA", "PFOA": "PF OA"}
    cf_pairs = []
    if args.cf_consist < 0 or not (0.0 < args.cf_prob <= 1.0):
        raise SystemExit(f"--cf-consist {args.cf_consist} must be >= 0 and "
                         f"--cf-prob {args.cf_prob} in (0, 1]")
    if args.cf_consist > 0:
        _cfb = [n for n, v in (("--sam-rho", args.sam_rho > 0),
                               ("--bagauc-gradcache",
                                bool(getattr(args, "bagauc_gradcache", False))),
                               ("--flip-swap", args.flip_swap > 0),
                               ("--bagauc-w", args.bagauc_w > 0))
                if v]
        if _cfb:
            # flip-swap would mirror the canonical frame for half the studies
            # (medial would become image-right); SAM/gradcache recompute the
            # loss outside the hooked block.
            raise SystemExit(f"--cf-consist is not isolated from {_cfb}")
        _ft = _CF_ALIAS.get(args.cf_target, args.cf_target)
        if _ft not in FINDINGS:
            raise SystemExit(f"--cf-target {args.cf_target!r} unknown")
        cf_pairs.append((FINDINGS.index(_ft), args.cf_mask))
        if args.cf_converse:
            _conv = {"Lateral OA": "Medial OA", "Medial OA": "Lateral OA"}
            if _ft not in _conv:
                raise SystemExit("--cf-converse needs --cf-target LatOA|MedOA")
            cf_pairs.append((FINDINGS.index(_conv[_ft]),
                             "lateral" if args.cf_mask == "medial"
                             else "medial"))
        print(f"[cfconsist] w={args.cf_consist} mode={args.cf_mode} "
              f"prob={args.cf_prob} pairs="
              f"{[(FINDINGS[j], h) for j, h in cf_pairs]} (Cor/Ax half-frame "
              f"zeroed in the canonical frame; medial = cols < W/2)",
              flush=True)
    cf_stat = {"n": 0, "sum": 0.0, "dz": 0.0, "pos": 0}
    _CF_VERBOSE = os.environ.get("CFCONSIST_VERBOSE", "") == "1"  # smoke-only log cadence
    if args.darc_local:
        _db = [n for n, v in (("--evidence-mil off", not args.evidence_mil),
                              ("--pat", bool(args.pat)),
                              ("--slicewit-targets",
                               bool(args.slicewit_targets))) if v]
        if _db:
            raise SystemExit(f"--darc-local is refused with {_db}")
        print("[darc] ON: TDE evidence, companion rows projected out of each "
              "non-Synovitis ev_head row (Synovitis row verbatim)", flush=True)

    # --- ev-adapter (scratch/evform/PREREG.md) ----------------------------
    # Refuse-not-skip. The adapter is PARAMETER-BEARING and sits on the serve
    # path, so a checkpoint carrying it can never be served by a consumer that
    # does not build it; the flags that replace or bypass the evidence readout
    # are refused here rather than silently ignored.
    if args.ev_adapter != "off":
        if not args.evidence_mil:
            raise SystemExit("--ev-adapter needs --evidence-mil: it acts on "
                             "the slice token the evidence readout consumes")
        _ea_bad = [n for n, v in (("--rubric-pool", bool(args.rubric_pool)),
                                  ("--pfrel", str(args.pfrel) != "off"
                                   if hasattr(args, "pfrel") else False),
                                  ("--bypass-study-transformer",
                                   bool(args.bypass_study_transformer))) if v]
        if _ea_bad:
            raise SystemExit(f"--ev-adapter is not isolated from {_ea_bad}: "
                             f"these replace or bypass the evidence readout "
                             f"the adapter feeds")
        if args.ev_adapter_k < 1 or args.ev_adapter_width < 1:
            raise SystemExit("--ev-adapter-k/--ev-adapter-width must be >= 1")

    # --- WITNESS-EV (scratch/witness/PREREG.md) ---------------------------
    # Refuse-not-skip, same reasoning as evdrop above: the witness KL is added
    # at exactly ONE site (the standard per-microstep study loss, right after
    # the primary soft-BCE) and it reads out[1] = the per-series evidence, so
    # every path that computes the study loss elsewhere or changes what out[1]
    # means is refused rather than silently half-applied.
    if args.witness_targets:
        if not args.evidence_mil:
            raise SystemExit("--witness-targets needs --evidence-mil: the "
                             "witness matches the model's PER-SERIES evidence "
                             "tensor, which only the EvidenceMIL aggregator "
                             "returns")
        if not os.path.exists(args.witness_targets):
            raise SystemExit(f"--witness-targets {args.witness_targets!r} "
                             f"does not exist")
        _wt_bad = [n for n, v in (("--bagauc-w", args.bagauc_w > 0),
                                  ("--sam-rho", args.sam_rho > 0),
                                  ("--label-repair-sidecar",
                                   bool(args.label_repair_sidecar)),
                                  ("--evidence-mil-v2",
                                   str(args.evidence_mil_v2) != "off"),
                                  ("--syno-specialist",
                                   bool(args.syno_specialist)),
                                  ("--auxbag", bool(args.auxbag)),
                                  ("--cutmix", args.cutmix > 0),
                                  ("--mmix-p", args.mmix_p > 0),
                                  ("--dismix-p", args.dismix_p > 0),
                                  ) if v]
        if _wt_bad:
            raise SystemExit(
                f"--witness-targets is not isolated from {_wt_bad}; the "
                f"witness KL is added at one site only and these change where "
                f"the study loss is computed or what out[1] contains "
                f"(--cutmix also mixes two studies' series into one item, "
                f"which would make the series->witness alignment a lie)")
        if args.witness_beta < 0.0 or not math.isfinite(args.witness_beta):
            raise SystemExit(f"--witness-beta must be a finite non-negative "
                             f"float, got {args.witness_beta!r}")
        if args.witness_beta == 0.0:
            print("[witness] WARNING beta=0: the witness term is computed and "
                  "logged but adds EXACTLY nothing to the loss (the "
                  "bit-for-bit control form)", flush=True)
        if not (args.witness_tau > 0.0) or not math.isfinite(args.witness_tau):
            raise SystemExit(f"--witness-tau must be a finite positive float, "
                             f"got {args.witness_tau!r}")
    _wit_ct = [0, 0.0, 0]          # [batches with a term, sum L, gated cells]
    # --- SLICE-WITNESS (scratch/slicewit/PREREG.md) -----------------------
    # Refuse-not-skip. Everything the witness refuses (one loss site, out[1]
    # meaning, cutmix), plus every flag that changes WHICH native slices fill
    # the 24 slots (the teacher is keyed by native slice index and the student's
    # index is replayed from the grid-jitter draw) or re-runs ev_head.
    _sw_ct = [0, 0.0, 0]           # [batches with a term, sum L, gated cells]
    if args.slicewit_targets:
        if not args.evidence_mil:
            raise SystemExit("--slicewit-targets needs --evidence-mil (per-slice "
                             "evidence exists only in the EvidenceMIL head)")
        if not os.path.exists(args.slicewit_targets):
            raise SystemExit(f"--slicewit-targets {args.slicewit_targets!r} "
                             f"does not exist")
        _sw_bad = [n for n, v in (("--bagauc-w", args.bagauc_w > 0),
                                  ("--sam-rho", args.sam_rho > 0),
                                  ("--label-repair-sidecar",
                                   bool(args.label_repair_sidecar)),
                                  ("--evidence-mil-v2",
                                   str(args.evidence_mil_v2) != "off"),
                                  ("--syno-specialist",
                                   bool(args.syno_specialist)),
                                  ("--auxbag", bool(args.auxbag)),
                                  ("--cutmix", args.cutmix > 0),
                                  ("--mmix-p", args.mmix_p > 0),
                                  ("--dismix-p", args.dismix_p > 0),
                                  ("--evdrop", bool(args.evdrop)),
                                  ("--stratified-slices",
                                   bool(args.stratified_slices)),
                                  ("--evidence-sampling-map",
                                   bool(args.evidence_sampling_map)),
                                  ("--native-ctx", bool(args.native_ctx)),
                                  ("--flip-swap", args.flip_swap > 0),
                                  ("--use-geometry", bool(args.use_geometry)),
                                  ("--aux-seg-w", args.aux_seg_w > 0),
                                  ("--grad-checkpoint",
                                   bool(args.grad_checkpoint)),
                                  ) if v]
        if _sw_bad:
            raise SystemExit(f"--slicewit-targets is not isolated from "
                             f"{_sw_bad}: these change the loss site, what "
                             f"ev_head sees, or which native slices fill the "
                             f"slots (the slice-witness alignment key)")
        for _nm, _v in (("beta", args.slicewit_beta), ("tau", args.slicewit_tau)):
            if not math.isfinite(_v) or _v < 0 or (_nm == "tau" and _v == 0):
                raise SystemExit(f"--slicewit-{_nm} invalid: {_v!r}")

    # --- auxbag (scratch/auxbag/PREREG.md) --------------------------------
    # Refuse-not-skip: both alternatives REPLACE the same per-cell BCE, so a
    # silent precedence rule would let a launch claim an objective it did not
    # train. Mutual exclusion is checked once, at startup, not per step.
    if args.auxbag and args.aux_visibility_mask:
        raise SystemExit(
            "--auxbag and --aux-visibility-mask both replace the per-series "
            "aux objective; refusing to guess which one this run meant")
    if args.auxbag and args.label_repair_sidecar:
        raise SystemExit(
            "--auxbag and --label-repair-sidecar both replace the per-series "
            "aux objective; refusing to guess which one this run meant")

    def _aux_bag_loss(aux_, y_, m_, wa_):
        """Bag-consistent aux likelihood: ONE term per (study, finding).

        q_sc = sigmoid(aux_), Q_c = 1 - prod_s (1 - q_sc) over the VALID
        series only; L_c = BCE(t_c, Q_c) / S_valid. The product is taken in
        log space (log(1-q) = logsigmoid(-a), never log(1-sigmoid) directly),
        and invalid series are multiplied by the existing series mask so they
        contribute log(1-q) = 0 and are exactly absent from the product.
        The 1/S_valid keeps the auxiliary term on the same scale as the
        per-cell mean it replaces, so --aux-w carries over unchanged.
        """
        w2 = wa_ if wa_.dim() == 2 else wa_.squeeze(1)      # (B,12)
        l1q = torch.nn.functional.logsigmoid(-aux_.float())  # log(1-q) (B,S,12)
        sv = m_.sum(1).clamp(min=1)                          # (B,1) valid series
        l1mQ = (l1q * m_).sum(1).clamp(max=-1e-6)            # log(1-Q)  (B,12)
        lQ = torch.log1p(-torch.exp(l1mQ))                   # log Q     (B,12)
        cell = -(y_ * lQ + (1.0 - y_) * l1mQ)                # BCE(t,Q)  (B,12)
        return (cell * w2 / sv).sum() / y_.shape[0] / y_.shape[1]

    def _label_repair_aux_loss(aux_, y_, m_, wa_, batch_, stype_):
        """Per-series classifier loss with the same repaired intervals."""
        lo = batch_["lr_lo"].to(aux_.device, non_blocking=True).unsqueeze(1)
        hi = batch_["lr_hi"].to(aux_.device, non_blocking=True).unsqueeze(1)
        interval = batch_["lr_interval"].to(
            aux_.device, non_blocking=True).unsqueeze(1)
        cells = label_repair_bce_elements(
            aux_, y_.unsqueeze(1).expand_as(aux_),
            lo.expand_as(aux_), hi.expand_as(aux_), interval.expand_as(aux_))
        if _vis_tab is not None:
            v = m_ * _vis_tab[stype_]
            _vis_acc[0] += float(v.sum())
            _vis_acc[1] += float(m_.sum()) * y_.shape[1]
            return (cells * v * wa_).sum() / v.sum().clamp(min=1)
        return ((cells * m_ * wa_).sum()
                / m_.sum().clamp(min=1) / y_.shape[1])
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"[seed] deterministic matched-arm seed {args.seed}", flush=True)
    else:
        torch.backends.cudnn.benchmark = True
    # torch.cuda.is_bf16_supported() answers True on sm_75 (T4) because it
    # counts EMULATED bf16, but cuDNN cannot plan a bf16 depthwise conv there:
    # timm's ConvNeXt conv_dw raised 'GET was unable to find an engine to
    # execute this computation' on node m3t100 (pfrel smoke 60154027). Real
    # bf16 tensor cores start at sm_80, so gate on the capability and fall back
    # to fp16 (the GradScaler below is already enabled for that dtype).
    # Production GPUs (L40S sm_89, A100 sm_80) are unaffected: still bf16.
    _cc = torch.cuda.get_device_capability()
    amp_dtype = (torch.bfloat16
                 if (_cc[0] >= 8 and torch.cuda.is_bf16_supported())
                 else torch.float16)
    print(f"[amp] sm_{_cc[0]}{_cc[1]} -> {amp_dtype}", flush=True)

    # Echo the resolved configuration. Every launcher default is env-overridable
    # (${TEXT_W:-0.2} and friends), so an override was previously invisible
    # except as a missing log line further down - you had to infer the setting
    # from which messages did NOT appear.
    # A queued job runs whatever the working tree contains when it STARTS, not
    # when it was submitted. The mute arm sat in the queue 4h15m and picked up
    # six commits, so it ran different code from its own control - and the
    # 5-fold spread across three versions - with nothing in any log to show it.
    try:
        import subprocess as _sp
        _h = _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                              cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
                              stderr=_sp.DEVNULL).decode().strip()
        _d = _sp.call(["git", "diff", "--quiet", "HEAD"],
                      cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        print(f"[code] git {_h}{' +dirty' if _d else ''}", flush=True)
    except Exception as _e:
        print(f"[code] git version unavailable: {type(_e).__name__}", flush=True)
    print("[args] " + "  ".join(f"{k}={v}" for k, v in sorted(vars(args).items())),
          flush=True)

    if args.no_gold:
        if not args.study_allowlist_npz:
            raise SystemExit("--no-gold requires --study-allowlist-npz")
        if args.calib != "fixed" or args.silent_calib != "none":
            raise SystemExit("--no-gold requires --calib fixed and "
                             "--silent-calib none")
        if args.gold_train_w != 0:
            raise SystemExit("--no-gold requires --gold-train-w 0")
        gold = {}
        print("[gold] DISABLED: labels, images, loaders and evaluation are not "
              "opened; internal pool comes from --study-allowlist-npz",
              flush=True)
    else:
        gold = load_gold()

    # Calibration must not see the studies we score on, or GOLD-AUC flatters
    # itself. Under "kfold" only the held-out gold slice is scored.
    calib_exclude, gold_eval = (), None
    if args.calib == "kfold":
        _fit_on, gold_eval = gold_calib_split(gold, args.calib_fold,
                                              args.calib_folds)
        calib_exclude = gold_eval
    prob_table, counts = fit_prob_table(args.labels, mode=args.calib,
                                        exclude=calib_exclude)
    print(f"[calib] mode={args.calib}  P(gold=1 | grade):",
          {g: round(p, 3) for g, p in sorted(prob_table.items())},
          " n:", dict(sorted(counts.items())), flush=True)
    # Ladder END values - the targets of a categorically-stated grade 0 or 3.
    # Read from the SAME table the targets were built with, never hardcoded:
    # the kxrq builder hardcoded [0.056...] (the gold-FITTED table) while
    # production runs --calib fixed ({0:0.05,...}), and its own mean-preserving
    # guard passed because it checked against the same wrong reference.
    _ladder_lo = float(prob_table[0])
    _ladder_hi = float(prob_table[3])
    if args.calib == "gold":
        print("[calib] WARNING fitted on all 58 gold studies - those same "
              "studies are the eval set, so GOLD-AUC below is optimistic",
              flush=True)
    elif args.calib == "kfold":
        print(f"[calib] fold {args.calib_fold}/{args.calib_folds}: scoring only "
              f"the {len(gold_eval)} gold studies excluded from the table",
              flush=True)

    silent = None
    if args.silent_calib == "gold":
        silent, s_counts = fit_silent_table(args.labels, pooled=prob_table.get(-1),
                                            k=args.silent_shrink,
                                            exclude=calib_exclude)
        if not calib_exclude:
            # calib_exclude is only non-empty under --calib kfold, so with the
            # launcher's default --calib fixed this table is fitted on all 58
            # gold studies and then scored on those same 58. The warning above
            # fires for --calib gold only, so this path used to leak silently.
            print("[silent] WARNING fitted on all 58 gold studies - those same "
                  "studies are the eval set, so GOLD-AUC below is optimistic. "
                  "Use --calib kfold for a held-out read, or judge this arm on "
                  "the leaderboard.", flush=True)
        # Reordering the target ladder is the only thing that can move a rank
        # metric, and per-finding silent rates almost never do: measured against
        # --calib fixed this changes the label ceiling by +0.0016 mean and
        # +0.000 on Synovitis, whose ladder stays 0.05 < 0.22 < 0.35 < 0.70.
        # So expect slower erosion toward the same asymptote, not a higher one.
        print("[silent] per-finding P(gold=1 | not mentioned), shrunk k="
              f"{args.silent_shrink}:", flush=True)
        for f, v in zip(FINDINGS, silent):
            print(f"   {f:<18} n={s_counts[f]:>3}  {v:.3f}", flush=True)
    labels = load_report_labels(args.labels, prob_table, silent)
    # Fracture negative-side SIGN FLIP (anomaly-miner find #1, 2026-08-13).
    # Measured over 88 arms: an explicit "no fracture" marks a HIGHER-risk
    # study than report silence - radiologists deny fracture precisely when
    # trauma context raises the question (contusion>=2 rate 14x, gold
    # P(frac|denial) 0.130 vs 0.067 silent) - yet the ladder teaches
    # denial(0.05) < silent(0.10) on 93% of the column, and 63/88 models
    # partially overrule their own target. This flips the order for the
    # REPORT-derived cells only: external-pool grades (e.g. fastMRI grade 0)
    # are verified expert absences, not conditional denials, and keep the
    # ladder. A rank CHANGE, not a monotone re-spacing - the silent-calib
    # kill does not cover it. Prior-driven (no gold tuning).
    if args.frac_neg_flip:
        import json as _json
        iF = FINDINGS.index("Fracture")
        _pool = set()
        if args.cotrain_pool:
            with open(args.cotrain_pool) as _f:
                for _v in _json.load(_f).values():
                    _pool |= set(_v)
        n_d = n_s = 0
        with open(args.labels) as _f:
            for _line in _f:
                _r = _json.loads(_line)
                _u = _r["uid"]
                if _u in _pool or _u not in labels:
                    continue
                _g = _r["grades"].get("Fracture", -1)
                if _g == 0:
                    labels[_u][iF] = 0.09
                    n_d += 1
                elif _g == -1:
                    labels[_u][iF] = 0.05
                    n_s += 1
        print(f"[frac-neg-flip] Fracture denial->0.09 on {n_d} studies, "
              f"silent->0.05 on {n_s}; external pool untouched", flush=True)
    # Self-training: replace the report-derived target with a blend of it and
    # the model's own out-of-fold prediction, per finding. Applied AFTER the
    # prob table so the two paths cannot disagree about calibration, and only
    # for studies the file names - anything absent keeps its report target.
    # Loaded here, APPLIED after the split (see below). Applying it here would
    # also rewrite the validation fold's targets - and with that fold's own OOF
    # predictions, produced by the very checkpoint that was selected on those
    # studies. Measured: scoring the previous model against blended val targets
    # lifts softAUCn 0.7204 -> 0.8037 for free, so checkpoint selection would
    # reward resembling the old model rather than being right.
    soft = None
    if args.soft_targets:
        st = np.load(args.soft_targets, allow_pickle=True)
        if list(str(x) for x in st["findings"]) != FINDINGS:
            raise SystemExit("--soft-targets finding order differs from FINDINGS")
        soft = ({str(u): r.astype(np.float32) for u, r in zip(st["uids"], st["targets"])},
                st["alpha"])
    # Delayed teacher supervision is deliberately separate from
    # --soft-targets. The latter is a static label replacement (and therefore
    # happens after the split); this map is read-only and consulted at each
    # live training step so persistent DataLoader workers cannot hold stale
    # labels and validation labels are never rewritten.
    teacher_targets = None
    teacher_source_meta = {}
    if args.teacher_curriculum != "off":
        if args.soft_targets:
            raise SystemExit("--teacher-curriculum is mutually exclusive with "
                             "--soft-targets; use one target intervention")
        if not args.teacher_targets:
            raise SystemExit("--teacher-curriculum requires --teacher-targets")
        if not (0.0 < args.teacher_max_w <= 1.0):
            raise SystemExit("--teacher-max-w must be in (0,1]")
        if args.teacher_curriculum == "late":
            if not (0.0 <= args.teacher_ramp_start < args.teacher_ramp_end):
                raise SystemExit("late teacher needs 0 <= ramp-start < ramp-end")
            if args.teacher_ramp_end > args.epochs:
                raise SystemExit("late teacher ramp-end cannot exceed --epochs")
        tz = np.load(args.teacher_targets, allow_pickle=True)
        if "uids" not in tz or "targets" not in tz or "findings" not in tz:
            raise SystemExit("--teacher-targets must contain uids, targets, findings")
        if list(str(x) for x in tz["findings"]) != FINDINGS:
            raise SystemExit("--teacher-targets finding order differs from FINDINGS")
        tu = [str(x) for x in tz["uids"]]
        tp = np.asarray(tz["targets"], dtype=np.float32)
        if tp.shape != (len(tu), len(FINDINGS)):
            raise SystemExit(f"--teacher-targets shape {tp.shape} does not match "
                             f"({len(tu)}, {len(FINDINGS)})")
        if len(set(tu)) != len(tu):
            raise SystemExit("--teacher-targets contains duplicate UIDs")
        if not np.isfinite(tp).all() or tp.min() < 0.0 or tp.max() > 1.0:
            raise SystemExit("--teacher-targets contains non-finite or out-of-range probabilities")
        gold_set = set(gold)
        gold_in_teacher = sorted(gold_set.intersection(tu))
        if gold_in_teacher:
            raise SystemExit("--teacher-targets contains gold-58 UIDs; refusing "
                             f"to train on expert evaluation rows ({len(gold_in_teacher)})")
        teacher_targets = {u: row for u, row in zip(tu, tp)}
        teacher_source_meta = {
            "path": os.path.abspath(args.teacher_targets),
            "uids": len(tu),
            "teacher": str(tz["teacher"]) if "teacher" in tz else "<absent>",
            "teacher_fold": int(tz["teacher_fold"]) if "teacher_fold" in tz else None,
            "members": int(len(tz["members"])) if "members" in tz else None,
        }
        # This sidecar metadata is copied into every checkpoint's args dict;
        # the source path and roster size then remain auditable after a run.
        args.teacher_source_meta = teacher_source_meta
        print(f"[teacher] {args.teacher_curriculum} source={args.teacher_targets} "
              f"uids={len(tu)} max_w={args.teacher_max_w:g} "
              f"ramp={args.teacher_ramp_start:g}->{args.teacher_ramp_end:g} "
              f"gold_overlap=0", flush=True)
    series_index = load_series_index(args.series_csv)
    if args.plane_filter:
        # Specialist member: restrict to one plane. Studies with nothing in
        # that plane keep their full list rather than being dropped, so the
        # study set - and therefore the MinHash grouping and the fold split -
        # is byte-identical to every other arm's. Comparability beats purity.
        kept, fell_back = {}, 0
        for s, lst in series_index.items():
            f = [t for t in lst if t[1] == args.plane_filter]
            if f:
                kept[s] = f
            else:
                kept[s] = lst
                fell_back += 1
        series_index = kept
        print(f"[plane] --plane-filter {args.plane_filter}: "
              f"{len(series_index) - fell_back} studies restricted, "
              f"{fell_back} had no {args.plane_filter} series and keep all",
              flush=True)
    lat = load_laterality(args.laterality_map)
    # Physical-scale resampling. This was live in the inference notebook while
    # train.py never passed geometry= at all, so the model was trained on
    # un-resampled volumes and scored on resampled ones - a silent train/serve
    # skew worth a mean 0.007-0.022 shift per prediction. The flag is written
    # into the checkpoint so inference can no longer disagree by omission.
    geom = load_geometry() if args.use_geometry else {}
    seqmap = load_sequence() if args.use_sequence else {}
    agree = {}
    if args.agree_weights:
        agree = load_agreement(args.agree_weights, args.agree_floor)
        if not agree:
            raise SystemExit(f"--agree-weights {args.agree_weights!r} loaded nothing")
        _a = np.stack(list(agree.values()))
        print(f"[agree] {len(agree)} studies, floor {args.agree_floor}; mean weight "
              + " ".join(f"{f.split()[0][:4]}:{v:.2f}"
                         for f, v in zip(FINDINGS, _a.mean(0))), flush=True)
    pairmax = {}
    if args.pair_max:
        import json as _json
        from data import PRIOR_TABLE
        pairmax = {u: PRIOR_TABLE[int(g)]
                   for u, g in _json.load(open(args.pair_max)).items()}
        if not pairmax:
            raise SystemExit(f"--pair-max {args.pair_max!r} loaded no studies")
        print(f"[pairmax] {len(pairmax)} studies, w={args.pair_max_w}", flush=True)
    comp_pairs = {}
    if args.comp_pair and os.path.exists(args.comp_pair):
        comp_pairs = {k: [tuple(p) for p in v]
                      for k, v in json.load(open(args.comp_pair)).items()}
        _n = sum(len(v) for v in comp_pairs.values())
        print(f"[comp-pair] {_n} constraints over {len(comp_pairs)} studies, "
              f"w={args.comp_pair_w}", flush=True)
    aux_lab, aux_names = ({}, [])
    if args.report_aux:
        aux_lab, aux_names = load_aux_labels(args.report_aux)
        if not aux_lab:
            raise SystemExit(f"--report-aux {args.report_aux!r} loaded no studies")
        print(f"[raux] {len(aux_lab)} studies x {len(aux_names)} fields, "
              f"w={args.report_aux_w}: {', '.join(aux_names)}", flush=True)
    # ---- MOAKS subregional aux targets (scratch/latoarescue/PREREG.md) ----
    # Flag-gated and ADDITIVE: with --moaks-aux absent every object below is
    # empty/None and no downstream branch fires - byte-path-identical to the
    # incumbent (PREREG code contract #2).
    moaks_lab, moaks_ho_lab, moaks_meta = {}, {}, {}
    moaks_names, moaks_swap = [], None
    moaks_block_names, moaks_block_slices = [], []
    if args.moaks_aux:
        from data import (MOAKS_AUX_BLOCKS, MOAKS_AUX_COLS,
                          MOAKS_AUX_FLIP_SWAP, load_moaks_aux)
        if args.report_aux:
            raise SystemExit("--moaks-aux and --report-aux are mutually "
                             "exclusive (both use the auxiliary-query slot "
                             "and out[-1])")
        if args.bagauc_w > 0 or args.sam_rho > 0:
            raise SystemExit("--moaks-aux is not implemented for the "
                             "--bagauc-w / --sam-rho loss paths (each owns "
                             "the loss assembly and neither is tested with "
                             "this term)")
        _all_moaks, moaks_meta = load_moaks_aux(args.moaks_aux, prob_table)
        if not _all_moaks:
            raise SystemExit(f"--moaks-aux {args.moaks_aux!r} loaded no rows")
        moaks_names = list(MOAKS_AUX_COLS)
        moaks_swap = MOAKS_AUX_FLIP_SWAP
        _off = 0
        for _bn, _bc in MOAKS_AUX_BLOCKS:
            moaks_block_names.append(_bn)
            moaks_block_slices.append(slice(_off, _off + len(_bc)))
            _off += len(_bc)
        assert _off == len(moaks_names)
        # KNEE-level holdout (PREREG amendment 1): all visits of one knee
        # (subject-side) land on the same side of the split, so V01 of a
        # held-out knee can never leak its V00 aux supervision. Held-out rows
        # keep their competition-label training role - only the aux sidecar
        # coverage differs, so the training uid set (and the folds) are
        # untouched.
        import hashlib as _hl
        _knee = lambda u: u.split("-", 1)[1]           # strip visit prefix
        _n_ho = max(int(args.moaks_aux_holdout), 0)
        _is_ho = (lambda u: _n_ho > 0 and int(
            _hl.md5(_knee(u).encode()).hexdigest(), 16) % _n_ho == 0)
        moaks_lab = {u: v for u, v in _all_moaks.items() if not _is_ho(u)}
        moaks_ho_lab = {u: v for u, v in _all_moaks.items() if _is_ho(u)}
        print(f"[moaksaux] {args.moaks_aux}: {len(_all_moaks)} knee-visit rows "
              f"x {len(moaks_names)} cols, w={args.moaks_aux_w}; blocks "
              + " ".join(f"{n}:{s.stop - s.start}" for n, s in
                         zip(moaks_block_names, moaks_block_slices))
              + f"; held out {len(moaks_ho_lab)} rows "
              f"({len({_knee(u) for u in moaks_ho_lab})} knees, knee-level "
              f"md5%{_n_ho}) from aux supervision for the learnability gate",
              flush=True)
    # ---- retanchor: within-run self-anchored function preservation --------
    # (scratch/retanchor/PREREG.md). Flag-gated and ADDITIVE: with the
    # default w=0.0 nothing below (nor any downstream ret_anchor branch)
    # runs - byte-path-identical to the incumbent, the same code-contract
    # standard as --moaks-aux / --bilat-rank. All checks fail in the first
    # minute, not at the end of epoch 5.
    if args.ret_anchor_w > 0:
        if args.ema_decay <= 0:
            raise SystemExit("--ret-anchor-w requires EMA (--ema-decay > 0): "
                             "the anchor IS the end-of-epoch EMA snapshot")
        if args.bagauc_w > 0 or args.sam_rho > 0:
            raise SystemExit("--ret-anchor-w is not implemented for the "
                             "--bagauc-w / --sam-rho loss paths (each owns "
                             "the loss assembly and neither is tested with "
                             "this term)")
        if args.cell_attn > 0 and args.cell_stage >= 0:
            raise SystemExit("--ret-anchor-w + --cell-attn/--cell-stage is "
                             "unsupported: the cell-stage forward hook "
                             "closes over the LIVE model instance, so a "
                             "deepcopied anchor would clobber the live "
                             "model's hooked features on every anchor "
                             "forward")
        if not (0 <= args.ret_anchor_epoch < args.epochs - 1):
            raise SystemExit(f"--ret-anchor-epoch {args.ret_anchor_epoch} "
                             f"leaves no post-anchor epoch inside --epochs "
                             f"{args.epochs}")
        print(f"[retanchor] w={args.ret_anchor_w}: anchor = end-of-ep"
              f"{args.ret_anchor_epoch} EMA snapshot; label-free soft-BCE "
              f"on the 12 challenge probs, EVERY batch of ep"
              f"{args.ret_anchor_epoch + 1}..{args.epochs - 1}", flush=True)
    if args.use_sequence:
        import collections as _c
        _d = _c.Counter(seqmap.values())
        print("[seq] " + "  ".join(f"{n}:{_d.get(i,0)}" for i, n in enumerate(SEQ_CLASSES))
              + f"  ({100*(len(seqmap)-_d.get(0,0))/max(len(seqmap),1):.1f}% resolved)",
              flush=True)
    print(f'[data] laterality known for {len(lat[0])} studies | '
          f'geometry {"ON, " + str(len(geom)) + " series" if geom else "OFF"}',
          flush=True)

    # The intensity window is a property of the VOLUME ROOT, not of this
    # command - train.py loads .npy files it did not create. Read it off the
    # root and stamp it into args so the checkpoint records which contract it
    # was trained on. Without this a re-windowed model is indistinguishable
    # from an old one, and the inference notebook (which re-implements the
    # transform) would feed it the wrong pixels with nothing raising.
    from preprocess import read_manifest
    args.pct_lo, args.pct_hi = read_manifest(args.vol_root)
    print(f"[data] intensity window p{args.pct_lo}-p{args.pct_hi} "
          f"(from {args.vol_root}"
          f"{'' if os.path.exists(os.path.join(args.vol_root, '_preprocess.json')) else ', LEGACY default'})",
          flush=True)

    # Only studies whose volumes actually exist on disk.
    have = {s for s in labels if os.path.isdir(os.path.join(args.vol_root, s))}
    # External co-training studies are held OUT of `studies` entirely, so every
    # downstream consumer of it - near_duplicate_groups, the group permutation,
    # the val fold - behaves exactly as on the competition pool alone. They are
    # added back to `tr` after the split, below.
    cotrain = set()
    if args.cotrain_pool:
        with open(args.cotrain_pool) as f:
            _cp = json.load(f)
        for _k, _v in sorted(_cp.items()):
            cotrain |= set(_v)
        _missing = len(cotrain - have)
        cotrain &= have
        if not cotrain:
            raise SystemExit(f"--cotrain-pool {args.cotrain_pool!r} matched no "
                             f"study with a volume under {args.vol_root!r}")
        print(f"[cotrain] {args.cotrain_pool}: {len(cotrain)} external studies "
              f"({_missing} listed without volumes); held out of the split",
              flush=True)
    if args.no_gold:
        with np.load(args.study_allowlist_npz, allow_pickle=False) as _az:
            if "uids" not in _az.files:
                raise SystemExit("--study-allowlist-npz lacks `uids`")
            _allow = [str(x) for x in _az["uids"].reshape(-1).tolist()]
        if len(_allow) != len(set(_allow)):
            raise SystemExit("--study-allowlist-npz contains duplicate UIDs")
        _missing_have = sorted(set(_allow) - have)
        _missing_labels = sorted(set(_allow) - set(labels))
        _external = sorted(set(_allow) & cotrain)
        if _missing_have or _missing_labels or _external:
            raise SystemExit(
                "invalid --study-allowlist-npz: "
                f"missing volumes={len(_missing_have)} "
                f"missing labels={len(_missing_labels)} "
                f"cotrain overlap={len(_external)}")
        studies = sorted(_allow)
        print(f"[data] no-Gold allowlist: {len(studies)} internal studies "
              f"from {args.study_allowlist_npz}", flush=True)
    else:
        studies = sorted(have - set(gold) - cotrain)

    # B3 gradient-provenance schedule: parse + validate BEFORE anything trains.
    cotrain_sched = None
    if args.cotrain_epochs:
        if not cotrain:
            raise SystemExit("--cotrain-epochs without --cotrain-pool gates "
                             "nothing - wrong invocation, refusing to train")
        try:
            _lo, _hi = (float(x) for x in args.cotrain_epochs.split(":"))
        except ValueError:
            raise SystemExit(f"--cotrain-epochs {args.cotrain_epochs!r} is not "
                             "LO:HI (floats)")
        if not (0 <= _lo < _hi):
            raise SystemExit(f"--cotrain-epochs {args.cotrain_epochs!r}: need "
                             "0 <= LO < HI")
        cotrain_sched = (_lo, _hi)
        print(f"[cotrain-sched] pool loss gated to epochs [{_lo}, {_hi}) of "
              f"{args.epochs}; loader/steps/LR identical to the constant arm",
              flush=True)

    # Gold studies as training rows. They carry EXPERT labels, not report-derived
    # ones, so they are the only rows whose targets are not a proxy. The held-out
    # slice never enters training and is the only slice scored below, so
    # GOLD-AUC stays honest - at the cost of being ~12 studies per fold, which
    # is why every fold has to be run and pooled before the number means much.
    gold_train, gold_hold = [], sorted(gold)
    if args.gold_train_w > 0:
        if args.gold_holdout_folds < 2:
            raise SystemExit("--gold-holdout-folds must be >= 2 (1 trains on "
                             "every gold study and leaves nothing to score on)")
        if not 0 <= args.gold_holdout_fold < args.gold_holdout_folds:
            raise SystemExit(f"--gold-holdout-fold {args.gold_holdout_fold} out "
                             f"of range for {args.gold_holdout_folds} folds; an "
                             "out-of-range value trains on all 58 and reports nan")
        if args.bs != 1:
            raise SystemExit(f"--gold-train-w needs --bs 1 (got {args.bs}): the "
                             "sample weight keys off b['study'][0], so at bs>1 "
                             "most gold rows go unweighted and non-gold rows in "
                             "the same batch get weighted instead")
        if args.limit:
            raise SystemExit("--limit truncates `studies` and would drop most of "
                             "the gold rows while still reporting them as added")
        _fit, gold_hold = gold_calib_split(gold, args.gold_holdout_fold,
                                           args.gold_holdout_folds)
        gold_train = [g for g in _fit if os.path.isdir(os.path.join(args.vol_root, g))]
        # Expert labels replace the report-derived targets for these studies —
        # unless --gold-train-labels report, the CONTROL arm of the
        # gold-substitution dose-response (scratch/goldsub/PREREG.md): same
        # rows, same upweight, report targets kept, so T−C isolates the label
        # SOURCE at fixed weighting and case mix.
        if args.gold_train_labels == "expert":
            # --gold-train-soft maps the expert BINARY (0/1) onto the ladder
            # ENDS read from prob_table (production: .05/.95), so a gold row's
            # target lives on the same scale every other training row does.
            # goldburn used hard 0/1 (default, unchanged). scratch/goldlite.
            for g in gold_train:
                gl = gold[g]
                if args.gold_train_soft:
                    gl = np.where(gl > 0.5, _ladder_hi,
                                  _ladder_lo).astype(np.float32)
                labels[g] = gl
            if args.gold_train_soft:
                print(f"[gold] expert targets mapped to ladder ends "
                      f"lo={_ladder_lo} hi={_ladder_hi}", flush=True)
        else:
            print("[gold] CONTROL arm: gold rows keep their REPORT-derived "
                  "targets (--gold-train-labels report)", flush=True)
        # Deliberately NOT added to `studies`. The near-duplicate grouping and
        # the RNG permutation that derives the val fold both key off this list,
        # so injecting 46 rows here changes the group COUNT and reshuffles every
        # fold: measured, val coverage fell to 2986/4344 with 1037 studies in
        # two folds and 1420 in none, and the val fold overlapped the baseline's
        # by ~10% Jaccard. The folds would stop being a partition, which breaks
        # OOF pooling and self-training. Gold rows are appended to `tr` after
        # the split instead.
        print(f"[gold] training on {len(gold_train)} gold studies at weight "
              f"{args.gold_train_w}; holding out {len(gold_hold)} for eval "
              f"(fold {args.gold_holdout_fold}/{args.gold_holdout_folds})",
              flush=True)

    # Report text, used twice below: to drop gold-bridging duplicates, and to
    # group the folds. Loaded once here so the two cannot disagree.
    import csv as _csv, hashlib
    _csv.field_size_limit(10 ** 7)
    with open(args.train_csv, newline="", encoding="utf-8") as f:
        rep = {r["StudyInstanceUID"]: " ".join(r["Report"].split())
               for r in _csv.DictReader(f)}
    rkey = lambda s: hashlib.md5(rep.get(s, s).encode()).hexdigest()

    # Drop training studies whose report text is byte-identical to a gold
    # study's. Their labels - and their auxiliary text target - are copies of a
    # study we evaluate on. Measured: 1 gold study, 4 training studies.
    if not args.no_gold and not args.keep_gold_dupes:
        # Unchanged by --gold-train-w on purpose. Narrowing this to just the
        # held-out slice would drop fewer studies, which changes `studies`,
        # which changes the near-duplicate group COUNT, which reshuffles the RNG
        # permutation and hands back a completely different validation fold -
        # measured, 844 val studies instead of 919, incomparable to every other
        # arm. The cost of leaving it alone is ~4 training rows that share text
        # with a gold-train study; the cost of touching it is the fold identity.
        # Gold rows are not in `studies` here, so they cannot be dropped by it.
        gold_texts = {rkey(s) for s in gold if s in rep}
        bridged = [s for s in studies if rkey(s) in gold_texts]
        if bridged:
            studies = [s for s in studies if s not in set(bridged)]
            print(f"[data] dropped {len(bridged)} training studies sharing a "
                  f"report with a gold study", flush=True)

    if args.limit:
        studies = studies[: args.limit]
    # When gold rows train, the eval set is ALWAYS their holdout - never the
    # --calib kfold slice, whose default 4 folds differ from the gold 5, so 12
    # of its 15 studies would be ones we trained on with hard expert labels.
    _gold_eval_set = (gold_hold if args.gold_train_w > 0
                      else (gold_eval if gold_eval is not None else gold))
    gold_studies = ([] if args.no_gold else
                    sorted(set(_gold_eval_set) & set(os.listdir(args.vol_root))))
    # A 12-study holdout AUC is not comparable to a 58-study one: measured on
    # real stored predictions, per-fold values span 0.137 for the SAME model,
    # and both documented ways of pooling folds are biased LOW (-0.014 mean-of-
    # folds, -0.040 rank-pooled). Tag it so no one lines the two up in a table.
    _gold_tag = ("NO-GOLD" if args.no_gold else
                 (f"GOLD{len(gold_studies)}-AUC"
                  if len(gold_studies) < 58 else "GOLD-AUC"))
    if args.no_gold:
        print(f"[data] train pool {len(studies)}   NO-GOLD evaluation disabled",
              flush=True)
    else:
        print(f"[data] train pool {len(studies)}   {_gold_tag} scored on "
              f"{len(gold_studies)} studies", flush=True)

    # Group by report text before splitting. 204 studies share byte-identical
    # reports (54 distinct texts, largest group 37), which means identical
    # derived labels; under a plain random split 27 of those 54 groups straddled
    # the boundary, so the fold score was partly measured on label vectors the
    # model had already fitted.
    # Near-duplicate grouping, not exact-hash. Institutional templates differ by
    # a sentence, so exact hashing calls them distinct while their derived labels
    # are identical: measured, 10.0% of a validation fold had a >=0.8 Jaccard
    # twin in train. MinHash at 0.8 cuts that to 0.7%. This matters twice - it
    # inflates the fold-AUC that selects checkpoints, and it contaminates the
    # out-of-fold predictions self-training will consume.
    gkey = near_duplicate_groups({s: rep.get(s, s) for s in studies},
                                 threshold=args.group_threshold)
    groups = sorted({gkey[s] for s in studies})
    rng = np.random.RandomState(args.split_seed)
    gperm = rng.permutation(len(groups))
    if args.split_seed != 42:
        print(f"[split] RE-DEAL: --split-seed {args.split_seed} (default 42). "
              f"Same {len(studies)} studies / {len(groups)} report-groups, a "
              f"DIFFERENT partition; fold-AUC is not comparable to a "
              f"default-deal run and OOF must not be pooled across deals.",
              flush=True)
    val_groups = {groups[i] for i in gperm[args.fold::args.n_folds].tolist()}
    tr = [s for s in studies if gkey[s] not in val_groups]
    va = [s for s in studies if gkey[s] in val_groups]

    # B1 label-corruption dose. Applied HERE - after the split, before any
    # cotrain/gold row joins - so that exactly three things are true: the val
    # fold is untouched (or the fold-AUC readout would measure the corruption
    # instead of the model), gold-58 rows are untouched (or the GOLD-AUC
    # readout would be corrupted too), and later joins add clean rows only.
    # Corruption is per (study, finding) CELL, drawing the replacement from
    # another training study's target for the same finding, which preserves
    # each column's marginal distribution and destroys only its alignment to
    # the study. The self-consistency AUC printed below (corrupted vs clean
    # target, over all train cells) is the measured x-axis for the
    # dose-response - the dose itself is only a knob, not the covariate.
    if args.label_noise > 0:
        assert args.label_noise < 1.0, "--label-noise must be in [0,1)"
        _nrng = np.random.RandomState(args.label_noise_seed)
        _cand = [u for u in tr if u in labels and u not in gold]
        _clean = {u: labels[u].copy() for u in _cand}
        _pool = np.stack([_clean[u] for u in _cand])          # (n_tr, 12)
        _hit = _nrng.rand(len(_cand), 12) < args.label_noise
        _don = _nrng.randint(0, len(_cand), size=(len(_cand), 12))
        for _i, _u in enumerate(_cand):
            _v = labels[_u].copy()
            for _j in range(12):
                if _hit[_i, _j]:
                    _v[_j] = _pool[_don[_i, _j], _j]
            labels[_u] = _v
        _new = np.stack([labels[u] for u in _cand])
        # soft_auc is O(n^2); a fixed 1500-study subsample keeps this diagnostic
        # under a second and its own sampling error near 0.003, far below the
        # dose separation it has to resolve.
        _sub = np.random.RandomState(0).choice(
            len(_cand), min(1500, len(_cand)), replace=False)
        _sc = [soft_auc(_pool[_sub, _j], _new[_sub, _j]) for _j in range(12)]
        print(f"[b1] label-noise dose {args.label_noise} seed "
              f"{args.label_noise_seed}: corrupted {int(_hit.sum())} of "
              f"{_hit.size} cells over {len(_cand)} train studies "
              f"({len(tr) - len(_cand)} gold/unlabelled studies untouched); "
              f"self-consistency AUC (corrupted vs clean) "
              f"{float(np.mean(_sc)):.4f}", flush=True)
    # Nested-CV teacher discipline (--drop-fold F): remove fold F's GROUPS from
    # the train half while validating on --fold as usual. A teacher trained
    # this way has never seen fold F, so its OOF targets can supervise a
    # student that validates on fold F without the stacking leak that voided
    # the first distillation round (v2b==clean on the LB). Group-level, same
    # permutation as the split, so the dropped set is exactly fold F.
    if args.drop_fold >= 0:
        if args.drop_fold == args.fold:
            raise SystemExit("--drop-fold equals --fold: the val fold is "
                             "already excluded from training; pick another")
        dg = {groups[i] for i in gperm[args.drop_fold::args.n_folds].tolist()}
        n0 = len(tr)
        tr = [s for s in tr if gkey[s] not in dg]
        print(f"[nested] --drop-fold {args.drop_fold}: dropped {n0 - len(tr)} "
              f"of {n0} train studies; val fold {args.fold} untouched", flush=True)
    # Leave-one-site-out drops the held manufacturer from the TRAIN half only.
    # The split above is untouched, so the val fold keeps its held-site studies
    # as the unseen-site probe and fold-AUC stays comparable to the control.
    if args.exclude_manufacturer:
        from site_shift_probe import load_site_meta
        smeta = load_site_meta()
        pat = args.exclude_manufacturer.lower()
        hit = lambda s: pat in smeta.get(s, {}).get("manufacturer", "").lower()
        n0 = len(tr)
        tr = [s for s in tr if not hit(s)]
        print(f"[site] --exclude-manufacturer '{args.exclude_manufacturer}': "
              f"dropped {n0 - len(tr)}/{n0} train studies; val keeps "
              f"{sum(1 for s in va if hit(s))} held-site studies as the "
              f"unseen-site probe", flush=True)
        if n0 == len(tr):
            raise SystemExit("[site] matched no training study - wrong substring?")
    # Gold rows join AFTER the split, so `studies`, the grouping and the val
    # fold are byte-identical to a run without this flag - fold-AUC and OOF stay
    # comparable to every other arm.
    _gset = set(gold_train)
    if _gset:
        tr = tr + sorted(_gset)
    # External pool joins here for the same reason gold does: after the split.
    # The val fold therefore contains ZERO external studies, which is what keeps
    # fold-AUC on the same case mix as every previous arm.
    if cotrain:
        tr = tr + sorted(cotrain)
        assert not (set(va) & cotrain), "external study leaked into validation"
    print(f"[data] fold {args.fold}: train {len(tr)} val {len(va)}  "
          f"(grouped on {len(groups)} distinct report texts)"
          + (f"  [+{len(cotrain)} external in train only]" if cotrain else ""),
          flush=True)

    # Audited label repairs are selected only now, after the validation split
    # and every training-only row join. The shared `labels` dict is never
    # rewritten: the training dataset applies these cells at __getitem__, while
    # validation retains both its original arrays and its original batch keys.
    label_repair = None
    if args.label_repair_sidecar:
        try:
            _all_repairs, _repair_meta = load_label_repair_sidecar(
                args.label_repair_sidecar, args.labels, gold_uids=gold,
                allow_unpromoted=args.allow_unpromoted_label_repair,
                promotion_path=args.label_repair_promotion or None)
            label_repair, _n_applied, _n_val_ignored = training_label_repairs(
                _all_repairs, tr, va)
        except (OSError, ValueError) as e:
            raise SystemExit(f"--label-repair-sidecar: {e}") from e
        if _n_applied == 0:
            raise SystemExit("--label-repair-sidecar matched no post-split "
                             "training cells")
        if (_repair_meta["actions"]["interval"]
                and (args.mmix_p > 0 or args.dismix_p > 0)):
            raise SystemExit("interval label repairs cannot be combined with "
                             "--mmix-p/--dismix-p: those losses mix point "
                             "targets and cannot preserve interval bounds")
        _repair_meta.update({
            "applied_entries": _n_applied,
            "applied_uids": len(label_repair),
            "validation_entries_ignored": _n_val_ignored,
        })
        if args.label_repair_expected_cells > 0 and \
                _n_applied != args.label_repair_expected_cells:
            raise SystemExit(
                f"--label-repair expected {args.label_repair_expected_cells} "
                f"cells but post-split application is {_n_applied}")
        if args.label_repair_receipt:
            _receipt = dict(_repair_meta,
                            train_uids=sorted(label_repair),
                            applied_uid_finding_cells=sorted(
                                f"{u}/{FINDINGS[k]}"
                                for u, cells in label_repair.items() for k in cells))
            os.makedirs(os.path.dirname(os.path.abspath(
                args.label_repair_receipt)), exist_ok=True)
            with open(args.label_repair_receipt, "w", encoding="utf-8") as _rf:
                json.dump(_receipt, _rf, indent=2, sort_keys=True)
                _rf.write("\n")
        args.label_repair_meta = _repair_meta
        print(f"[labelrepair] {_repair_meta['path']}: "
              f"{_n_applied}/{_repair_meta['entries']} cells on "
              f"{len(label_repair)} TRAIN studies; ignored "
              f"{_n_val_ignored} validation cells; gold overlap 0; sha256 "
              f"{_repair_meta['sha256'][:12]}", flush=True)

    # Per-study, per-finding loss coverage. Only the TRAIN half gets it: the
    # validation fold must keep full-weight report targets or fold-softAUCn
    # stops being comparable to every previous run. Same discipline as
    # --soft-targets below, and for the same reason.
    evidence_intervals = None
    evidence_mapped_uids = set()
    if args.evidence_sampling_map:
        try:
            evidence_intervals = load_evidence_sampling_map(
                args.evidence_sampling_map)
        except (OSError, ValueError) as e:
            raise SystemExit(f"--evidence-sampling-map: {e}") from e
        if not label_repair:
            raise SystemExit("--evidence-sampling-map requires a training "
                             "label-repair sidecar")
        evidence_mapped_uids = {
            key.split("/", 1)[0] for key in evidence_intervals
        }
        missing_evidence = sorted(set(label_repair) - evidence_mapped_uids)
        if missing_evidence:
            raise SystemExit(
                "--evidence-sampling-map lacks verified support for "
                f"{len(missing_evidence)} accepted training UIDs; first="
                f"{missing_evidence[:3]}")
        if args.label_repair_expected_cells > 0 and \
                len(evidence_mapped_uids) != args.label_repair_expected_cells:
            raise SystemExit(
                "--evidence-sampling-map mapped UID count "
                f"{len(evidence_mapped_uids)} differs from expected accepted "
                f"cell count {args.label_repair_expected_cells}")
        if args.label_repair_receipt:
            try:
                with open(args.label_repair_receipt, encoding="utf-8") as _rf:
                    _receipt = json.load(_rf)
            except (OSError, json.JSONDecodeError) as e:
                raise SystemExit(
                    f"--evidence-sampling-map: cannot update label-repair "
                    f"receipt: {e}") from e
            _receipt["evidence_sampling"] = {
                "map_path": os.path.abspath(args.evidence_sampling_map),
                "mapped_uid_count": len(evidence_mapped_uids),
                "mapped_uids": sorted(evidence_mapped_uids),
                "accepted_uid_count": len(label_repair),
                "unmapped_accepted_uids": [],
                "inference_sampling_unchanged": True,
            }
            with open(args.label_repair_receipt, "w", encoding="utf-8") as _rf:
                json.dump(_receipt, _rf, indent=2, sort_keys=True)
                _rf.write("\n")
        print(f"[evidence] verified native-cache intervals: "
              f"{len(evidence_intervals)} series; training-only sampling; "
              "inference sampling unchanged", flush=True)
    cover = None
    if args.coverage:
        d = np.load(args.coverage, allow_pickle=True)
        cover = {str(u): c.astype(np.float32)
                 for u, c in zip(d["uids"], d["coverage"])}
        n_tr = sum(1 for u in tr if u in cover)
        if n_tr == 0:
            raise SystemExit(f"--coverage {args.coverage!r} matched no TRAINING "
                             f"study - wrong file, or the pool was never added "
                             f"to the train split?")
        part = np.mean([cover[u] for u in tr if u in cover], axis=0)
        print(f"[coverage] {args.coverage}: {n_tr}/{len(tr)} train studies "
              f"masked; mean weight per finding "
              + " ".join(f"{f.split()[0][:4]}:{v:.2f}"
                         for f, v in zip(FINDINGS, part)), flush=True)

    # Freeze the eligible row set after the split. The teacher may have
    # predictions for validation studies (the existing T_max file is not a
    # nested-CV teacher), but those rows are never consulted by the training
    # loop. Keeping this set explicit makes the gold/validation boundaries
    # visible in the log and prevents a future batch-path change from silently
    # turning a teacher prediction into a validation target.
    teacher_train_uids = set()
    if teacher_targets is not None:
        teacher_train_uids = set(tr).intersection(teacher_targets)
        teacher_val_overlap = set(va).intersection(teacher_targets)
        if not teacher_train_uids:
            raise SystemExit("--teacher-targets matched no TRAINING study")
        if set(gold_train).intersection(teacher_train_uids):
            raise SystemExit("teacher target set intersects gold training rows")
        print(f"[teacher] eligible TRAIN rows {len(teacher_train_uids)}/{len(tr)}; "
              f"validation predictions ignored {len(teacher_val_overlap)}; "
              f"gold-train overlap 0", flush=True)

    # scratch/witness/PREREG.md: with --witness-targets the dataset class is
    # the UID-reporting subclass (one extra string key per item, no pixel,
    # label, sampling or RNG difference); off, this is KneeStudyDataset and the
    # construction below is byte-identical to every prior arm.
    _DS_CLS = WitnessStudyDataset if args.witness_targets else KneeStudyDataset
    if args.slicewit_targets:     # scratch/slicewit: + native slice index key
        _DS_CLS = SliceWitStudyDataset
    mk = lambda st, train: _DS_CLS(
        st, series_index, labels, args.vol_root, args.n_series,
        args.n_slices, args.size, train, laterality=lat, geometry=geom,
        sequence=seqmap,
        aux_labels=(moaks_lab if args.moaks_aux else aux_lab),
        aux_flip_swap=moaks_swap,           # None -> module default
        agree=agree,
        flip_swap=args.flip_swap, cutmix=args.cutmix,
        cutmix_alpha=args.cutmix_alpha,
        coverage=cover if train else None,
        pairmax=pairmax if train else None,
        domain_rand=args.domain_rand,
        vendor_aug=args.vendor_aug,
        series_dropout=args.series_dropout,
        spatial_aug=args.spatial_aug,
        affine_p=args.affine_p,
        aux_seg_root=(args.aux_seg_root
                     if (train and args.aux_seg_w > 0) else ""),
        pfrel_series=(_pfrel_mode != "off"), bone_series=(_bone_mode != "off"), bone_fluid=(0 if args.bone_series_fluid == "nonfs" else 1), native_ctx=args.native_ctx, slice_ctx=args.slice_ctx, ctx_mm=args.ctx_mm, zspacing=(load_zspacing() if args.ctx_mm > 0 else None), aspect=(load_aspect() if args.aspect_fix else None), full_fov=args.full_fov, xfuse_pairs=_xf_pairs,
        stratified_slices=args.stratified_slices if train else False,
        label_repair=label_repair if train else None,
        evidence_intervals=evidence_intervals if train else None,
        evidence_sampling=bool(args.evidence_sampling_map) if train else False)
    dl = lambda ds, shuf: DataLoader(
        ds, batch_size=args.bs, shuffle=shuf, num_workers=args.workers,
        pin_memory=True, drop_last=shuf, persistent_workers=args.workers > 0)

    # Self-training targets go ONLY to the training half. The validation fold
    # keeps its untouched report targets, so fold-softAUCn stays comparable to
    # the control run and cannot be inflated by the blend.
    if soft is not None:
        smap, alpha = soft
        n_tr = sum(1 for u in tr if u in smap)
        n_va = sum(1 for u in va if u in smap)
        for u in tr:
            if u in smap:
                labels[u] = smap[u]
        print(f"[soft] {args.soft_targets}: blended {n_tr}/{len(tr)} TRAIN targets; "
              f"left {n_va} val studies on report targets; "
              + " ".join(f"{f.split()[0][:4]}:{v:.2f}"
                         for f, v in zip(FINDINGS, alpha)), flush=True)
        if n_tr == 0:
            raise SystemExit("--soft-targets matched no TRAINING study - wrong file?")
        # The blend runs over `tr`, which now contains the gold rows. Restore
        # their expert labels: a self-training pseudo-label must never overwrite
        # ground truth. (Under --gold-train-labels report the run's convention
        # is report targets on these rows; restoring EXPERT labels here would
        # silently convert a control arm into a treatment arm.)
        if args.gold_train_labels == "expert":
            for g in gold_train:
                labels[g] = gold[g]

    # --distill-logit-mse COVERAGE ASSERTION (scratch/effwave/PREREG.md, effL).
    # An "activation print" is not enough: this project has twice shipped a
    # flag that logged itself as on while reaching zero cells (the roster
    # name-mismatch drops, the --mute-after unresolved-uid shape). So the exact
    # row set the term will touch is computed HERE, from the same objects the
    # blend used, and an empty set is a hard abort rather than a silent no-op.
    # The set is `tr` INTERSECT the soft-target file, MINUS any gold row whose
    # expert label was just restored above - those rows carry ground truth, not
    # a teacher probability, and pulling the student's logit toward
    # logit(0.95) on them would be a different (and unregistered) experiment.
    distill_uids = set()
    if args.distill_logit_mse > 0:
        if soft is None:
            raise SystemExit("--distill-logit-mse needs --soft-targets: there "
                             "is no teacher probability to match without one")
        if args.bagauc_w > 0 or args.sam_rho > 0:
            raise SystemExit("--distill-logit-mse is not implemented for the "
                             "--bagauc-w / --sam-rho loss paths (each owns the "
                             "loss and neither is tested with this term)")
        distill_uids = {u for u in tr if u in soft[0]}
        if args.gold_train_labels == "expert":
            distill_uids -= set(gold_train)
        if not distill_uids:
            raise SystemExit(
                "--distill-logit-mse: ZERO training studies carry a "
                "distillation target - the term would be an exact no-op while "
                "the flag logged itself as active. Wrong --soft-targets file?")
        print(f"[logitmse] w={args.distill_logit_mse} on "
              f"{len(distill_uids)}/{len(tr)} TRAIN studies x {len(FINDINGS)} "
              f"findings = {len(distill_uids) * len(FINDINGS)} cells; target = "
              f"logit(clip(p_teacher, 1e-4, 1-1e-4)); ADDED to soft-BCE",
              flush=True)

    # --aux-seg-w COVERAGE ASSERTION, same discipline as --distill-logit-mse's
    # above: an empty coverage set must hard-abort, not log itself active
    # while touching zero rows.
    if args.aux_seg_w > 0:
        if not args.aux_seg_root:
            raise SystemExit("--aux-seg-w needs --aux-seg-root")
        if not os.path.isdir(args.aux_seg_root):
            raise SystemExit(f"--aux-seg-w: --aux-seg-root "
                             f"{args.aux_seg_root} does not exist")
        _auxseg_cov = sum(1 for u in tr
                          if os.path.isdir(os.path.join(args.aux_seg_root, u)))
        if _auxseg_cov == 0:
            raise SystemExit(
                "--aux-seg-w: ZERO training studies have a mask directory "
                "under --aux-seg-root - the term would be an exact no-op "
                "while the flag logged itself as active.")
        print(f"[auxseg] w={args.aux_seg_w} root={args.aux_seg_root}: "
              f"{_auxseg_cov}/{len(tr)} TRAIN studies have a mask dir "
              f"(per-series availability decided per __getitem__ call)",
              flush=True)

    tr_dl = dl(mk(tr, True), True)
    va_dl = dl(mk(va, False), False)
    gold_dl = None
    if not args.no_gold:
        gold_ds = KneeStudyDataset(gold_studies, series_index, gold, args.vol_root,
                                   args.n_series, args.n_slices, args.size, False,
                                   laterality=lat, geometry=geom, sequence=seqmap,
                                   pfrel_series=(_pfrel_mode != "off"), bone_series=(_bone_mode != "off"), bone_fluid=(0 if args.bone_series_fluid == "nonfs" else 1), native_ctx=args.native_ctx, slice_ctx=args.slice_ctx, ctx_mm=args.ctx_mm, zspacing=(load_zspacing() if args.ctx_mm > 0 else None), aspect=(load_aspect() if args.aspect_fix else None), full_fov=args.full_fov, xfuse_pairs=_xf_pairs)
        gold_dl = dl(gold_ds, False)

    # Held-out-OAI eval loader for the MOAKS aux learnability gate (PREREG 4a)
    # + the amendment-1 OA-head prediction dump. Eval-mode dataset over the
    # held-out KNEES that actually train (holdout ∩ cotrain pool), carrying
    # the held-out rows' aux targets so y_aux rides along.
    moaks_val_dl = None
    if args.moaks_aux:
        _n_sup = sum(1 for u in tr if u in moaks_lab)
        if _n_sup == 0:
            raise SystemExit("--moaks-aux: ZERO training studies carry MOAKS "
                             "aux targets - the term would be a no-op while "
                             "the flag logs itself active. Wrong sidecar or "
                             "pool?")
        _mho = sorted(set(moaks_ho_lab) & cotrain)
        print(f"[moaksaux] {_n_sup}/{len(tr)} train studies carry aux targets; "
              f"holdout eval set {len(_mho)} studies (held-out knees in the "
              f"cotrain pool)", flush=True)
        if args.moaks_aux_holdout > 0:
            if not _mho:
                raise SystemExit("--moaks-aux: the knee-level holdout matched "
                                 "no cotrain-pool study - the learnability "
                                 "gate would be unreadable")
            moaks_val_dl = dl(KneeStudyDataset(
                _mho, series_index, labels, args.vol_root, args.n_series,
                args.n_slices, args.size, False, laterality=lat, geometry=geom,
                sequence=seqmap, aux_labels=moaks_ho_lab,
                pfrel_series=(_pfrel_mode != "off"), bone_series=(_bone_mode != "off"), bone_fluid=(0 if args.bone_series_fluid == "nonfs" else 1), native_ctx=args.native_ctx, slice_ctx=args.slice_ctx, ctx_mm=args.ctx_mm, zspacing=(load_zspacing() if args.ctx_mm > 0 else None), aspect=(load_aspect() if args.aspect_fix else None), full_fov=args.full_fov, xfuse_pairs=_xf_pairs), False)

    # Auxiliary report-text target (see src/embed_reports.py).
    text_vec, text_dim = {}, 0
    if args.text_emb and os.path.exists(args.text_emb) and args.text_w > 0:
        z = np.load(args.text_emb, allow_pickle=True)
        text_vec = {str(u): v for u, v in zip(z["uids"], z["emb"])}
        text_dim = int(z["emb"].shape[1])
        print(f"[text] {len(text_vec)} report embeddings, dim {text_dim}, "
              f"weight {args.text_w}", flush=True)

    model = KneeNet(args.backbone, text_dim=text_dim, stack_depth=args.stack_depth, study_pool=args.study_pool,
                    bypass_study_tr=args.bypass_study_transformer,
                    drop_path=args.drop_path,
                    slice_ctx=args.slice_ctx,
                    xfuse=args.xfuse,
                    n_seq=len(SEQ_CLASSES) if args.use_sequence else 0,
                    n_report_aux=len(aux_names),
                    moaks_blocks=tuple(s.stop - s.start
                                       for s in moaks_block_slices),
                    band_pool=args.band_pool,
                    routing_prior=args.routing_prior,
                    flat_attn=args.flat_attn,
                    flat_queries=([FINDINGS.index(n.strip()) for n in args.flat_queries.split(",") if n.strip()]
                                  if args.flat_queries else None),
                    cell_attn=args.cell_attn,
                    cell_stage=args.cell_stage,
                    gate_init=args.gate_init,
                    cell_slice_bands=args.cell_slice_bands,
                    gem_p=args.gem_pool,
                    osseous_factor=args.osseous_factor,
                    oa_gate=args.oa_gate,
                    query_family=args.query_family,
                    slice_enc=args.slice_enc,
                    n_slice_pos=args.n_slices if args.slice_pos != "none" else 0,
                    slice_pos_types=6 if args.slice_pos == "type" else 0,
                    aux_seg_classes=(6 if args.aux_seg_w > 0 else 0),
                    slice3d=args.slice3d,
                    blurpool=args.blurpool,
                    qsa=args.qsa, qsa_gate_init=args.qsa_gate_init,
                    evidence_mil=args.evidence_mil,
                    evdrop=args.evdrop, evdrop_p=args.evdrop_p,
                    pat=args.pat, pat_grid=args.pat_grid,
                    darc_local=args.darc_local,
                    ev_adapter=args.ev_adapter,
                    ev_adapter_k=args.ev_adapter_k,
                    ev_adapter_width=args.ev_adapter_width,
                    rubric_pool=args.rubric_pool,
                    evmil_prior=args.evmil_prior,
                    tfcrop_types=(4 if args.tfcrop_root else 0),
                    evmil_tau_init=args.evmil_tau_init,
                    evmil_dif_init=args.evmil_dif_init,
                    evidence_mil_v2=args.evidence_mil_v2,
                    ev2_gamma_init=args.ev2_gamma_init,
                    syno_specialist=args.syno_specialist,
                    syno_gamma_init=args.syno_gamma_init,
                    pfrel=_pfrel_mode,
                    pfrel_lambda_pos=args.pfrel_lambda_pos,
                    pfrel_beta=args.pfrel_beta,
                    bone=_bone_mode, bone_k=args.bone_k,
                    bone_tau=args.bone_tau,
                    bone_shell=args.bone_shell, ps25=args.ps25,
                    bn_valid_only=getattr(args, "bn_valid_only", False),
                    seq_adapt=getattr(args, "seq_adapt", 0)).to(device)
    if _bone_mode != "off":
        _bn = sum(p.numel() for n, p in model.named_parameters()
                  if n.startswith("bone_net."))
        print(f"[bone] arm={_bone_mode} new params={_bn} k={args.bone_k} "
              f"tau={args.bone_tau} shell={args.bone_shell} "
              f"aux_w={args.bone_aux} slices={args.n_slices}", flush=True)
        if args.bone_init:
            _isd = torch.load(args.bone_init, map_location="cpu")
            _isd = _isd["model"] if isinstance(_isd, dict) and "model" in _isd else _isd
            _bb = {k: v for k, v in _isd.items() if k.startswith("backbone.")}
            _miss = model.load_state_dict(_bb, strict=False)
            _n_ok = len(_bb) - len([k for k in _miss.unexpected_keys])
            if _n_ok == 0:
                raise SystemExit(
                    f"--bone-init {args.bone_init}: zero backbone tensors "
                    f"loaded; the trunk would train from the timm init while "
                    f"the log claimed a warm start")
            import hashlib as _hl
            with open(args.bone_init, "rb") as _f:
                _sha = _hl.sha256(_f.read()).hexdigest()
            print(f"[bone] init {args.bone_init} sha256={_sha} "
                  f"backbone tensors loaded={_n_ok}/{len(_bb)} "
                  f"unexpected={len(_miss.unexpected_keys)}", flush=True)
    if _pfrel_mode != "off":
        _pn = sum(p.numel() for n, p in model.named_parameters()
                  if n.startswith("pfrel_net."))
        print(f"[pfrel] arm={_pfrel_mode} new params={_pn} "
              f"lambda_pos={args.pfrel_lambda_pos} beta={args.pfrel_beta} "
              f"aux_w={args.pfrel_aux} slices={args.n_slices}", flush=True)
        if args.pfrel_init:
            _isd = torch.load(args.pfrel_init, map_location="cpu")
            _isd = _isd["model"] if isinstance(_isd, dict) and "model" in _isd else _isd
            _bb = {k: v for k, v in _isd.items() if k.startswith("backbone.")}
            _miss = model.load_state_dict(_bb, strict=False)
            _n_ok = len(_bb) - len([k for k in _miss.unexpected_keys])
            if _n_ok == 0:
                raise SystemExit(
                    f"--pfrel-init {args.pfrel_init}: zero backbone tensors "
                    f"loaded; the trunk would train from the timm init while "
                    f"the log claimed a warm start")
            import hashlib as _hl
            with open(args.pfrel_init, "rb") as _f:
                _sha = _hl.sha256(_f.read()).hexdigest()
            print(f"[pfrel] init {args.pfrel_init} sha256={_sha} "
                  f"backbone tensors loaded={_n_ok}/{len(_bb)} "
                  f"unexpected={len(_miss.unexpected_keys)}", flush=True)
    if getattr(args, "trauma_head", False):
        model._enable_trauma_head()
        print("[trauma_head] ON: P(C)=E*K, P(F)=E*L; K params "
              f"{sum(p.numel() for n, p in model.named_parameters() if n.startswith('trauma_k_'))}", flush=True)
    model.want_seg = args.aux_seg_w > 0
    # --- evdrop column weights + startup print (scratch/evdrop/PREREG.md) --
    _evdrop_cw = None
    if args.evdrop:
        if not hasattr(model, "ev_focal"):
            raise SystemExit("--evdrop: this model exposes no ev_focal "
                             "partition; refusing to train a flag that does "
                             "nothing")
        _evdrop_cw = torch.zeros(len(FINDINGS), device=device)
        _evdrop_cw[model.ev_focal] = 1.0
        print(f"[evdrop] ON: p={args.evdrop_p} unit=top-evidence SLICE "
              f"(argmax over the {args.n_slices} 3-tap-smoothed per-slice "
              f"evidences, one per study x series x focal finding), "
              f"masked-view loss on the FOCAL columns only, weight 1.0, "
              f"joint with the full-view loss; TRAIN-ONLY (identity at eval, "
              f"state dict unchanged, no kernel port)", flush=True)
        print("[evdrop] focal " + " ".join(
            f"{f}" for f in model.EVMIL_FOCAL), flush=True)
        print("[evdrop] untouched " + " ".join(
            f"{f}" for f in model.EVMIL_DIFFUSE), flush=True)
    # --- PAT column weights + startup print (scratch/shortcutarms §PAT) ---
    _pat_cw = None
    if args.pat:
        _pat_cw = torch.ones(len(FINDINGS), device=device)
        _pat_cw[FINDINGS.index("Synovitis")] = 0.0
        print(f"[pat] ON: grid={args.pat_grid} slabs/series of "
              f"{args.n_slices // args.pat_grid} slots, w={args.pat_w}, "
              f"aggregate=per-finding softmax over patches (self form), "
              f"columns=11 non-Synovitis (Synovitis weight 0 = placebo)",
              flush=True)
    # --- WITNESS-EV teacher (scratch/witness/PREREG.md) --------------------
    # The npz holds, per study, the per-series evidence of the honest-OOF EV
    # parent member that HELD THAT STUDY OUT, over EVERY eligible series (the
    # train loader's subsample is not known in advance), in the same space as
    # the student's out[1] (osseous_factorize already applied, as the model
    # applies it). Loaded once into a {study: (uid->row)} index; the alignment
    # to the batch's actual series happens per step.
    witness = None
    _wit_cols = None
    if args.witness_targets:
        _wz = np.load(args.witness_targets, allow_pickle=True)
        _wev = _wz["ev"].astype(np.float32)
        _wser = [str(s) for s in _wz["ser"]]
        _woff = _wz["offs"].astype(np.int64)
        _wfold = {str(_u): int(_f) for _u, _f in
                  zip(_wz["uids"], _wz["sfold"])}
        witness = {}
        for _i, _u in enumerate(_wz["uids"]):
            _a0, _a1 = int(_woff[_i]), int(_woff[_i + 1])
            witness[str(_u)] = ({_wser[_j]: _j for _j in range(_a0, _a1)}, _wev)
        # --- OUTER-FOLD ISOLATION (refuse-not-skip) ------------------------
        # Every study's witness must come from the EV parent member that HELD
        # THAT STUDY OUT. The file records, per study, the fold of the member
        # that produced it (`sfold`), and each per-fold extraction ran over
        # exactly that fold's held-out UID set (asserted against
        # scratch/t1oof/fold_holdouts.npz at extraction time). So the check
        # here is that the file's split AGREES with this run's own split: no
        # training study may carry a witness from THIS fold's member (that
        # member trained on this run's val half, and a training study whose
        # witness came from it would be a study that member also trained on),
        # and every val-half study that carries a witness must carry this
        # fold's. A disagreement means the two splits are not the same object.
        _iso_tr = [_u for _u in tr
                   if _u in _wfold and _wfold[_u] == args.fold]
        _iso_va = [_u for _u in va
                   if _u in _wfold and _wfold[_u] != args.fold]
        if _iso_tr or _iso_va:
            raise SystemExit(
                f"--witness-targets FOLD ISOLATION FAILURE: "
                f"{len(_iso_tr)} TRAIN studies carry a witness from this run's "
                f"own fold {args.fold} (e.g. {_iso_tr[:3]}) and {len(_iso_va)} "
                f"val studies carry a witness from another fold (e.g. "
                f"{_iso_va[:3]}). The witness file's split is not this run's "
                f"split; refusing to train on a leaked teacher.")
        if _wev.shape[1] != len(FINDINGS):
            raise SystemExit(f"--witness-targets: evidence has "
                             f"{_wev.shape[1]} columns, expected "
                             f"{len(FINDINGS)}")
        # Synovitis is EXCLUDED: it is the placebo column of this arm and the
        # served EV block carries weight 0 on it, so it must receive exactly
        # zero witness gradient.
        _syno = FINDINGS.index("Synovitis")
        _wit_cols = torch.ones(len(FINDINGS), device=device)
        _wit_cols[_syno] = 0.0
        _wit_ncol = len(FINDINGS) - 1
        _wit_n_tr = sum(1 for _u in tr if _u in witness)
        _wit_n_va = sum(1 for _u in va if _u in witness)
        print(f"[witness] ON beta={args.witness_beta} tau={args.witness_tau} "
              f"gate={args.witness_gate} "
              f"cells={_wit_n_tr}x{len(FINDINGS) - 1} "
              f"(studies with a witness: {_wit_n_tr}/{len(tr)} TRAIN, "
              f"{_wit_n_va}/{len(va)} val-rows-unused; "
              f"{len(witness)} in the file, {len(_wser)} series); "
              f"Synovitis EXCLUDED; TRAIN-ONLY (no parameter, no state-dict "
              f"key, identity at eval); outer-fold isolation VERIFIED "
              f"(0 TRAIN studies witnessed by this fold's own member)",
              flush=True)
        if _wit_n_tr == 0:
            raise SystemExit("--witness-targets matched no TRAINING study")
    # --- SLICE-WITNESS teacher (scratch/slicewit/PREREG.md) ---------------
    slicewit = None
    _sw_cols = None
    if args.slicewit_targets:
        slicewit, _sw_fold = load_slicewit_teacher(args.slicewit_targets,
                                                   len(FINDINGS))
        _iso_tr = [_u for _u in tr if _sw_fold.get(_u, -1) == args.fold]
        _iso_va = [_u for _u in va
                   if _u in _sw_fold and _sw_fold[_u] != args.fold]
        if _iso_tr or _iso_va:
            raise SystemExit(
                f"--slicewit-targets FOLD ISOLATION FAILURE: {len(_iso_tr)} "
                f"TRAIN studies carry a slice witness from this run's own fold "
                f"{args.fold} (e.g. {_iso_tr[:3]}), {len(_iso_va)} val studies "
                f"from another fold (e.g. {_iso_va[:3]}); refusing")
        _sw_syno = FINDINGS.index("Synovitis")
        _sw_cols = torch.tensor([_k for _k in range(len(FINDINGS))
                                 if _k != _sw_syno], dtype=torch.long,
                                device=device)
        _sw_n_tr = sum(1 for _u in tr if _u in slicewit)
        print(f"[slicewit] ON beta={args.slicewit_beta} tau={args.slicewit_tau} "
              f"gate=entropy_conf cols={len(_sw_cols)} (Synovitis EXCLUDED); "
              f"studies with a slice witness: {_sw_n_tr}/{len(tr)} TRAIN; "
              f"keyed by NATIVE slice index, renormalised over the shared "
              f"slices; TRAIN-ONLY (ev_head forward hook, no parameter/"
              f"buffer/state-dict key); outer-fold isolation VERIFIED",
              flush=True)
        if _sw_n_tr == 0:
            raise SystemExit("--slicewit-targets matched no TRAINING study")
    if args.auxbag:
        # --- auxbag bias init (scratch/auxbag/PREREG.md) -------------------
        # At bias 0 every q_sc ~ .5, so the bag probability
        # Q_c = 1 - .5^S ~ .97 at the modal S=5 for EVERY finding and the
        # objective starts saturated against the negatives. Initialise the
        # per-series logit at b_c = logit(1 - (1 - pi_c)^(1/5)), pi_c = the
        # TRAINING-fold prevalence of t_c >= .5, so the initial bag
        # probability equals that prior at S=5. Training rows only: the
        # held-out fold never touches this number.
        _nf = len(FINDINGS)
        _pi = np.zeros(_nf, np.float64)
        _npos = 0
        for _u in tr:
            _t = labels.get(_u)
            if _t is None:
                continue
            _pi += (np.asarray(_t, np.float32)[:_nf] >= 0.5)
            _npos += 1
        if _npos == 0:
            raise SystemExit("--auxbag: no training row carries a label "
                             "vector; the prevalence bias init is undefined")
        _pi = np.clip(_pi / _npos, 1e-3, 1.0 - 1e-3)
        _q0 = 1.0 - (1.0 - _pi) ** (1.0 / 5.0)
        _b0 = torch.from_numpy(np.log(_q0 / (1.0 - _q0)).astype(np.float32))
        _b0 = _b0.to(next(model.parameters()).device)  # 60314706 died: cpu vs cuda
        with torch.no_grad():
            if getattr(model, "aux_head", None) is not None:
                model.aux_head[-1].bias[:_nf].copy_(_b0)
                _where = "aux_head[-1].bias"
            elif getattr(model, "ev_head", None) is not None:
                # EvidenceMIL: out[1] is the per-series evidence ep, and the
                # study logit is ev_b + sum_s r_s * ep_s with sum_s r_s = 1.
                # The slice LSE and the mean/quantile-.8/max mixture are both
                # shift-equivariant, so adding b_c to ev_head's bias shifts
                # ep by exactly b_c and the study logit by exactly b_c.
                # Subtracting b_c from ev_b therefore leaves the STUDY-logit
                # init bit-identical to the control's: this arm differs from
                # its matched control in the auxiliary objective alone.
                model.ev_head[-1].bias[:_nf].copy_(_b0)
                model.ev_b[:_nf].sub_(_b0)
                _where = "ev_head[-1].bias (ev_b compensated, study init "
                _where += "unchanged)"
            else:
                raise SystemExit("--auxbag: this model exposes no per-series "
                                 "aux logit head to initialise")
        print(f"[auxbag] ON: bag aux likelihood replaces the broadcast "
              f"per-series BCE; aux_w={args.aux_w} unchanged; "
              f"init {_where}; train rows={_npos}", flush=True)
        print("[auxbag] pi_c   " + " ".join(
            f"{f[:4]}={p:.4f}" for f, p in zip(FINDINGS, _pi)), flush=True)
        print("[auxbag] bias_c " + " ".join(
            f"{f[:4]}={float(v):+.3f}" for f, v in zip(FINDINGS, _b0)),
            flush=True)
    if args.qsa != "none":
        _qn = sum(p.numel() for n, p in model.named_parameters()
                  if ".qsa_" in n)
        print(f"[qsa] mode={args.qsa}: {_qn} new params, "
              f"gate_init={float(model.study_enc.qsa_gate.item()):.3f}",
              flush=True)
    if args.slice3d != "none":
        _s3 = [(n, p.numel()) for n, p in model.named_parameters()
               if ".conv_dw.prev." in n or ".conv_dw.next." in n]
        print(f"[slice3d] mode={args.slice3d}: {len(model._slice3d_mods)} "
              f"blocks wrapped, {sum(k for _, k in _s3)} new params in "
              f"{len(_s3)} tensors", flush=True)
    # --- regbundle (scratch/regbundle/PREREG.md) ---------------------------
    # Manifold mixup is train-only state on the model (want_seg precedent).
    # The trainer-side bank (previous study's raw target / cell weight /
    # sharpen) lives here and is advanced in lock-step with the model's z
    # bank inside the primary-loss block. mmix_p=0 executes no extra op.
    model.mmix_p = float(args.mmix_p)
    model.mmix_alpha = float(args.mmix_alpha)
    model._mmix_rng = random.Random(None if args.seed < 0 else args.seed + 1013)
    mmix_bank_y = mmix_bank_w = mmix_bank_sh = None
    mmix_steps = 0
    if args.mmix_p > 0:
        # Every other loss that reads `logits`/`y` AFTER the primary soft-BCE
        # would see the MIXED tensors; none is preregistered with mmix, so
        # refuse rather than silently mix them.
        # Amendment 2 (scratch/regbundle/PREREG.md): pair_max_w / comp_pair_w
        # are NOT in this list any more. Both are WITHIN-study terms
        # (pairmax = max of the two meniscus logits of one study; comp_pairs
        # = softplus(z_lo - z_hi) inside one study) and both read `logits`
        # AFTER the primary-loss block has restored the UNMIXED tensor
        # (`y, _w, _sh, logits = _y_raw, _w_orig, _sh_raw, _lg_raw`), so
        # they are computed exactly as in the reference recipe; only the
        # primary soft-BCE ever sees the mixed logits / targets.
        _mm_bad = [k for k in ("distill_logit_mse", "bagauc_w",
                               "rank_w", "aucm",
                               "rep_gls_sharp", "ret_anchor_w", "gram_escrow_w")
                   if float(getattr(args, k, 0) or 0) != 0]
        if _mm_bad:
            raise SystemExit(f"--mmix-p > 0 is not isolated from {_mm_bad}; "
                             f"mixing is implemented for the primary loss only")
        print(f"[regbundle] mmix coexists with within-study pair losses on "
              f"UNMIXED logits: pair_max_w={args.pair_max_w} "
              f"comp_pair_w={args.comp_pair_w}", flush=True)
    print(f"[regbundle] blurpool={args.blurpool} "
          f"({len(model._blurpool_mods)} blurs installed) "
          f"mmix_p={args.mmix_p} alpha={args.mmix_alpha}", flush=True)
    # --- dismix (scratch/dismix/PREREG.md) ---------------------------------
    # Ordinal-adjacent per-finding query mixup. dismix_p=0 (every pre-existing
    # arm): `dismix` stays None, the model never stashes, no extra op.
    dismix = None
    if args.dismix_p > 0:
        _dm_bad = [k for k in ("mmix_p", "distill_logit_mse", "bagauc_w",
                               "rank_w", "aucm", "ret_anchor_w",
                               "gram_escrow_w")
                   if float(getattr(args, k, 0) or 0) > 0]
        if _dm_bad:
            raise SystemExit(f"--dismix-p > 0 is not isolated from {_dm_bad}; "
                             "only the primary soft-BCE may see mixed tensors")
        from dismix import DisMix
        dismix = DisMix(ladder=[prob_table[g] for g in (0, 1, 2, 3)],
                        n_out=len(FINDINGS), p=args.dismix_p,
                        alpha=args.dismix_alpha, max_age=args.dismix_max_age,
                        seed=(None if args.seed < 0 else args.seed + 2027))
        model.dismix_on = True
        print(f"[dismix] p={args.dismix_p} alpha={args.dismix_alpha} "
              f"max_age={args.dismix_max_age} ladder="
              f"{[round(float(prob_table[g]), 3) for g in (0, 1, 2, 3)]} "
              f"(silent {round(float(prob_table[-1]), 3)} never mixes)",
              flush=True)
    if args.backbone_ckpt:
        # Domain-SSL continued-pretraining init (scratch/sslpre/). Loaded over
        # the timm-created backbone AFTER construction, so the architecture and
        # every other module keep their standard init; strict=True because a
        # partial load here would silently train a half-random backbone.
        _bb = torch.load(args.backbone_ckpt, map_location="cpu")
        _sd = _bb.get("backbone", _bb)
        model.backbone.load_state_dict(_sd, strict=True)
        print(f"[backbone-ckpt] loaded {len(_sd)} tensors from "
              f"{args.backbone_ckpt} (ssl step {_bb.get('step', '?')})",
              flush=True)
    # ---- full-model resume (scratch/frztrunk/PREREG.md) -------------------
    # AFTER --backbone-ckpt on purpose: a full resume subsumes it, and if both
    # were ever passed the resume must win rather than half of it be overwritten.
    if args.resume_model:
        _rk = torch.load(args.resume_model, map_location="cpu",
                         weights_only=False)
        _rsd = _rk["model"] if isinstance(_rk, dict) and "model" in _rk else _rk
        _ra = _rk.get("args") if isinstance(_rk, dict) else None
        if _ra is not None:
            # Same contract list the surgery assay gates on, plus the flags
            # that are PARAMETER-FREE and therefore invisible to a strict
            # state-dict load (gem_pool, osseous_factor) - the exact class of
            # setting that has silently mismatched twice in this project.
            _g = ((lambda k: _ra.get(k)) if isinstance(_ra, dict)
                  else (lambda k: getattr(_ra, k, None)))
            # A checkpoint written BEFORE a contract flag existed simply has no
            # such key, and the old gate read that absence as the value None and
            # refused every resume from an older snapshot (forktest ft_eq
            # 60137105 died on bypass_study_transformer: None vs False). Absence
            # is not a mismatch: the run that wrote the checkpoint necessarily
            # used the argparse DEFAULT for a flag it did not have. So an ABSENT
            # key is compared against ap.get_default(key) and equality is a
            # match, logged rather than silent; a key that is PRESENT and
            # disagrees is still refused, unchanged.
            _absent = ((lambda k: k not in _ra) if isinstance(_ra, dict)
                       else (lambda k: not hasattr(_ra, k)))
            for _c in ("backbone", "n_series", "n_slices", "size",
                       "laterality_map", "use_geometry", "use_sequence",
                       "osseous_factor", "oa_gate", "slice_enc", "slice_ctx",
                       "stack_depth", "study_pool",
                       "bypass_study_transformer", "syno_specialist",
                       "cotrain_pool", "report_aux", "query_family", "qsa",
                       "gem_pool", "band_pool", "cell_attn", "cell_stage",
                       "flat_attn", "labels", "vol_root", "series_csv",
                       "fold", "n_folds"):
                _mine = getattr(args, _c, None)
                if getattr(args, "resume_init_only", False) and _c in (
                        "labels", "vol_root", "series_csv", "cotrain_pool", "laterality_map"):
                    print(f"[resume-init-only] data-path key {_c} exempt: ckpt "
                          f"{_g(_c)!r} vs run {_mine!r}", flush=True)
                    continue
                if _absent(_c):
                    _dflt = ap.get_default(_c)
                    if _dflt != _mine:
                        raise SystemExit(
                            f"--resume-model contract mismatch on {_c}: "
                            f"checkpoint ABSENT (default {_dflt!r}) vs this "
                            f"run {_mine!r}")
                    print(f"[resume-model] contract: {_c} absent in ckpt, "
                          f"default {_dflt!r} matches", flush=True)
                    continue
                if _g(_c) != _mine:
                    raise SystemExit(
                        f"--resume-model contract mismatch on {_c}: checkpoint "
                        f"{_g(_c)!r} vs this run {_mine!r}")
        model.load_state_dict(_rsd, strict=True)
        print(f"[resume-model] loaded {len(_rsd)} tensors STRICT from "
              f"{args.resume_model} (epoch={_rk.get('epoch')} "
              f"epoch_float={_rk.get('epoch_float')} ema={_rk.get('ema')} "
              f"subepoch={_rk.get('subepoch')})", flush=True)

    # ---- rubricrung per-rung heads (default off: nothing below runs) -------
    rung_rows, rung_tgt = [], {}
    if args.rung_heads:
        for _part in args.rung_heads.split(";"):
            _f, _ks = _part.split(":")
            for _k in _ks.split(","):
                rung_rows.append((int(_f), int(_k)))
        assert all(0 <= f < len(FINDINGS) and 1 <= k <= 3 for f, k in rung_rows), rung_rows
        assert len(set(rung_rows)) == len(rung_rows), rung_rows
        model.enable_rung_heads([f for f, _ in rung_rows])
        _lo, _hi = float(prob_table[0]), float(prob_table[3])
        _nr = 0
        with open(args.labels, encoding="utf-8") as _f_:
            for _line in _f_:
                _r = json.loads(_line)
                if not ("grades" in _r and "probs" in _r and "evidence" in _r):
                    continue
                _t = np.zeros(len(rung_rows), np.float32)
                _m = np.zeros(len(rung_rows), np.float32)
                for _j, (_fi, _k) in enumerate(rung_rows):
                    _g = int(_r["grades"].get(FINDINGS[_fi], -1))
                    if _g >= 0:
                        _t[_j], _m[_j] = (_hi if _g >= _k else _lo), 1.0
                    elif _k >= 2:
                        _t[_j], _m[_j] = _lo, 1.0
                rung_tgt[_r["uid"]] = (_t, _m)
                _nr += 1
        _pos = np.sum([v[0] > 0.5 for v in rung_tgt.values()], 0)
        print(f"[rung] rows {[(FINDINGS[f], k) for f, k in rung_rows]} "
              f"lo/hi {_lo}/{_hi} w={args.rung_w} report rows {_nr} "
              f"positives per row {_pos.tolist()}", flush=True)

    # ---- parameter freezing ------------------------------------------------
    frozen_params = []
    if args.freeze:
        _want = {x.strip() for x in args.freeze.split(",") if x.strip()}
        _bad = _want - {"backbone", "trunkrest", "readout"}
        if _bad:
            raise SystemExit(f"--freeze: unknown partition(s) {sorted(_bad)}; "
                             f"choose from backbone,trunkrest,readout")

        def _partition(n):
            if n.startswith("backbone."):
                return "backbone"
            if (n in ("head", "head_b", "study_enc.queries")
                    or n.startswith("study_enc.route.")
                    or n.startswith("study_enc.norm.")):
                return "readout"
            if n.startswith(("series_enc.", "study_enc.tr.",
                             "study_enc.type_emb.", "aux_head.")):
                return "trunkrest"
            # Loud, not silent: an unclassified parameter would otherwise be
            # trained by default and the "frozen trunk" would have a hole in it.
            raise SystemExit(f"--freeze: parameter {n!r} is in no partition; "
                             f"extend _partition() before freezing a model "
                             f"with this flag set")
        _counts = {}
        for n, p in model.named_parameters():
            c = _partition(n)
            _counts[c] = _counts.get(c, 0) + 1
            if c in _want:
                p.requires_grad_(False)
                frozen_params.append((n, p))
        if args.freeze_expect:
            for tok in args.freeze_expect.split(","):
                k, _, v = tok.partition("=")
                k, v = k.strip(), int(v)
                if _counts.get(k, 0) != v:
                    raise SystemExit(
                        f"--freeze-expect {k}={v} but the built model has "
                        f"{_counts.get(k, 0)}; partition counts {_counts}")
            print(f"[freeze] --freeze-expect {args.freeze_expect} VERIFIED",
                  flush=True)
        _nf = sum(p.numel() for _, p in frozen_params)
        _nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[freeze] partition counts {_counts}; FROZEN {sorted(_want)} = "
              f"{len(frozen_params)} tensors / {_nf/1e6:.2f}M params; "
              f"TRAINABLE {sum(1 for p in model.parameters() if p.requires_grad)}"
              f" tensors / {_nt/1e6:.4f}M params", flush=True)
        print("[freeze] trainable: " + ", ".join(
            n for n, p in model.named_parameters() if p.requires_grad),
            flush=True)

    if args.slice_pos != "none":
        # The seq_emb arm burned a full 5.5-hour run being silently inert at
        # 50x too small a scale, and the only reason that was ever caught was
        # measuring the output shift. Do it here, before the run starts.
        with torch.no_grad():
            pv = model.series_enc.pos
            print(f"[slice-pos] mode={args.slice_pos}  table {tuple(pv.shape)}  "
                  f"std {pv.std():.3f}  norm/slice {pv.norm(dim=-1).mean():.2f} "
                  f"(slice tokens measure ~29.6, so this is "
                  f"{float(pv.norm(dim=-1).mean()) / 29.6:.0%} of a token)",
                  flush=True)
    # ---- LoRA on a frozen trunk (scratch/dinofam/PREREG.md, dinolora arm) ----
    # Additive: with --lora-rank 0 and no --freeze-backbone this block is a no-op.
    if args.lora_rank > 0 or args.freeze_backbone:
        import lora as _lora
        if args.lora_rank > 0:
            if not args.freeze_backbone:
                raise SystemExit("--lora-rank > 0 requires --freeze-backbone "
                                 "(the arm is frozen trunk + adapters by design)")
            _nb, _np_ = _lora.inject_lora(model.backbone, args.lora_rank, args.lora_alpha)
            print(f"[lora] rank {args.lora_rank} alpha {args.lora_alpha} on q,v of "
                  f"{_nb} attention blocks: {_np_/1e6:.3f}M adapter params", flush=True)
        _nf, _ne, _nt = _lora.freeze_backbone(model, keep_lora=True)
        print(f"[lora] backbone frozen: {_nf} tensors / {_ne/1e6:.1f}M params; "
              f"trainable backbone params {_nt/1e6:.3f}M; total trainable "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M",
              flush=True)
    if args.grad_checkpoint:
        try:
            model.backbone.set_grad_checkpointing(True)
            print("[mem] gradient checkpointing ON for the backbone", flush=True)
        except Exception as e:
            raise SystemExit(f"--grad-checkpoint unsupported by {args.backbone}: {e}")
    bb_named, rest, nodecay = [], [], []
    evad = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            # --freeze: kept OUT of the optimizer entirely, not merely
            # requires_grad=False. AdamW's decoupled weight decay is applied
            # inside step() to every param it holds that has a grad, and a
            # frozen param that ever received a stale grad would be shrunk
            # toward zero with no gradient to oppose it. Exclusion also frees
            # the two Adam moment buffers per frozen tensor.
            continue
        if n.startswith("ev_adapter_mod."):
            # --ev-adapter (scratch/evform): its OWN optimiser group at
            # --ev-adapter-lr (default 5 x head LR). Split BEFORE the
            # backbone/head/no-decay branches so the group is exactly the
            # adapter, whatever --no-decay-norm-bias or --llrd are doing.
            evad.append((n, p))
            continue
        if n.startswith("backbone.") and ".lora_" not in n:
            # LoRA adapters (".lora_" only ever appears with --lora-rank > 0)
            # go to `rest`: adapter LR = head LR, per the dinolora prereg.
            bb_named.append((n, p))
        elif n == "series_enc.pos":
            # A positional table under weight decay is pulled toward zero, i.e.
            # toward the permutation-invariant model this flag exists to leave.
            # It would arrive at inert SLOWLY, which is worse than arriving
            # there at init: the run still costs five hours and the [slice-pos]
            # scale line at startup would look healthy the whole time.
            nodecay.append(p)
        else:
            rest.append(p)
    nd_bb, nd_rest = [], []
    if args.no_decay_norm_bias:
        # rb arm: pull every 1-D tensor (norm weight/bias, LayerScale gamma,
        # Linear/Conv bias) out of the decayed groups BEFORE the LLRD/flat
        # split so the groups below stay pure backbone / pure head (the
        # --wd-from mixing assertion relies on that purity).
        nd_bb = [(n, p) for n, p in bb_named if p.ndim <= 1]
        bb_named = [(n, p) for n, p in bb_named if p.ndim > 1]
        nd_rest = [p for p in rest if p.ndim <= 1]
        rest = [p for p in rest if p.ndim > 1]
    if args.llrd < 1.0:
        # Layer-wise LR decay: the flat backbone multiplier is the one schedule
        # confound every non-convnext architecture verdict shares. Depth
        # buckets: stem=0, stages.N=N+1, remaining backbone params (final
        # norm etc.) = top. Top bucket gets lr*backbone_lr_mult, each step
        # down multiplies by llrd - so --llrd 1.0 reproduces the flat schedule.
        # Two depth maps, because the technique and the implementation were
        # aimed at different families. LLRD is a TRANSFORMER fine-tuning
        # technique - early blocks want a far lower LR than late ones - but
        # this only ever matched ConvNeXt's stages.N, so it could only be
        # tested on the architecture where it is least likely to help. Its
        # measured null on convnext_tiny is therefore not a verdict on LLRD;
        # it is a verdict on LLRD-as-a-ConvNeXt-recipe, which is all that
        # could run. ViTs bucket per BLOCK, so a 12-layer ViT-B gets 14 groups
        # against ConvNeXt's 6 - the finer granularity is the point.
        _stage = lambda n: re.match(r"backbone\.stages\.(\d+)\.", n)     # convnext
        _block = lambda n: re.match(r"backbone\.blocks\.(\d+)\.", n)     # vit
        _unit = _stage if any(_stage(n) for n, _ in bb_named) else _block
        sids = [int(_unit(n).group(1)) for n, _ in bb_named if _unit(n)]
        if not sids:
            raise SystemExit(f"--llrd: no backbone.stages.N or backbone.blocks.N "
                             f"parameters in {args.backbone}; extend the depth "
                             f"map before using LLRD with this backbone family")
        n_stages = max(sids) + 1
        # The ViT stem is patch_embed plus the two token tables; cls_token and
        # pos_embed are leaves directly under backbone., so they would other-
        # wise fall through to the TOP bucket and be trained at the highest
        # backbone LR - the exact opposite of what LLRD is for.
        _STEM = ("backbone.stem", "backbone.patch_embed",
                 "backbone.cls_token", "backbone.pos_embed")
        def _depth(n):
            if n.startswith(_STEM):
                return 0
            m = _unit(n)
            return int(m.group(1)) + 1 if m else n_stages + 1
        buckets = {}
        for n, p in bb_named:
            buckets.setdefault(_depth(n), []).append(p)
        top = max(buckets)
        groups = [{"params": ps,
                   "lr": args.lr * args.backbone_lr_mult * args.llrd ** (top - d)}
                  for d, ps in sorted(buckets.items())]
        max_lr = [g["lr"] for g in groups]
        print(f"[llrd] {len(groups)} backbone depth groups, lr {max_lr[0]:.2e} "
              f"-> {max_lr[-1]:.2e} (decay {args.llrd})", flush=True)
        groups.append({"params": rest, "lr": args.lr})
        max_lr.append(args.lr)
    else:
        groups = [{"params": [p for _, p in bb_named],
                   "lr": args.lr * args.backbone_lr_mult},
                  {"params": rest, "lr": args.lr}]
        max_lr = [args.lr * args.backbone_lr_mult, args.lr]
    if nodecay:
        groups.append({"params": nodecay, "lr": args.lr, "weight_decay": 0.0})
        max_lr.append(args.lr)
        print(f"[slice-pos] {sum(p.numel() for p in nodecay)} params excluded "
              f"from weight decay", flush=True)
    if args.no_decay_norm_bias:
        _ndb = [p for _, p in nd_bb]
        groups.append({"params": _ndb, "lr": args.lr * args.backbone_lr_mult,
                       "weight_decay": 0.0})
        max_lr.append(args.lr * args.backbone_lr_mult)
        groups.append({"params": nd_rest, "lr": args.lr, "weight_decay": 0.0})
        max_lr.append(args.lr)
        _sz = lambda ps: f"{len(ps)} tensors / {sum(p.numel() for p in ps)} params"
        print(f"[no-decay-norm-bias] DECAYED (wd {args.wd}): backbone "
              f"{_sz([p for _, p in bb_named])}, head {_sz(rest)} | "
              f"NO-DECAY (ndim<=1): backbone {_sz(_ndb)}, head {_sz(nd_rest)}",
              flush=True)
    if any(not g["params"] for g in groups):
        # --freeze can empty a whole group (freezing the backbone empties the
        # backbone group). OneCycleLR's max_lr list is positional, so the two
        # lists have to be filtered together.
        _keep = [i for i, g in enumerate(groups) if g["params"]]
        print(f"[freeze] dropping {len(groups)-len(_keep)} empty param "
              f"group(s); {len(_keep)} remain", flush=True)
        groups = [groups[i] for i in _keep]
        max_lr = [max_lr[i] for i in _keep]
    # --ev-adapter (scratch/evform/PREREG.md): the adapter gets its OWN group
    # at --ev-adapter-lr (default 5 x head LR), appended AFTER the freeze
    # filter so its max_lr entry can never be dropped with an empty group.
    # Split decayed (ndim>1: U, V, W_route) from no-decay (ndim<=1: the shared
    # LayerNorm and the router bias), mirroring --no-decay-norm-bias.
    if evad:
        _ea_lr = float(args.ev_adapter_lr) or (args.lr * 5.0)
        _ea_d = [p for n, p in evad if p.ndim > 1]
        _ea_n = [p for n, p in evad if p.ndim <= 1]
        for _ps, _wd in ((_ea_d, args.wd), (_ea_n, 0.0)):
            if _ps:
                groups.append({"params": _ps, "lr": _ea_lr,
                               "weight_decay": _wd})
                max_lr.append(_ea_lr)
        print(f"[ev-adapter] {args.ev_adapter} k={args.ev_adapter_k} "
              f"width={args.ev_adapter_width} "
              f"eff_width={getattr(model.ev_adapter_mod, 'width', None)} "
              f"params={sum(p.numel() for _, p in evad)} "
              f"({len(_ea_d)} decayed tensors wd={args.wd}, {len(_ea_n)} "
              f"no-decay) lr={_ea_lr:.3e} (head lr {args.lr:.3e})", flush=True)
    elif args.ev_adapter != "off":
        raise SystemExit("--ev-adapter is on but no ev_adapter_mod.* parameter "
                         "reached the optimizer; refusing to train an inert arm")

    # ACTIVITY CHECKS (scratch/evform/PREREG.md, read BEFORE any AUC): the
    # adapter must have MOVED off its function-preserving init and the router
    # must not have collapsed. Snapshot the init here; _evform_activity()
    # prints the delta and the mean router entropy (max = log K).
    _ea_init = ({n: p.detach().clone() for n, p in evad} if evad else None)

    def _evform_activity(where=""):
        if _ea_init is None:
            return
        import math as _m
        d = [float((p.detach() - _ea_init[n]).norm()) for n, p in evad]
        n0 = [float(_ea_init[n].norm()) for n, _ in evad]
        pi = getattr(model, "_ev_route_pi", None)
        ent = tstd = float("nan")
        if pi is not None:
            _p = pi.float().reshape(-1, pi.shape[-1])
            ent = float(-(_p * _p.clamp_min(1e-9).log()).sum(-1).mean())
            tstd = float(_p.std(0).max())
        emax = _m.log(pi.shape[-1]) if pi is not None else float("nan")
        print(f"[evform-act]{where} adapter |dW| total "
              f"{sum(d):.4e} (init norm {sum(n0):.4e}, rel "
              f"{sum(d)/max(sum(n0), 1e-9):.4e}); per-tensor "
              f"{[round(x, 5) for x in d]}; router entropy {ent:.4f} of "
              f"max {emax:.4f}; pi std over tokens (max col) {tstd:.4e}",
              flush=True)
    # --pcol-w (scratch/recipe2): learned per-grade prototypes live OUTSIDE the
    # model (checkpoints unchanged) in their own no-decay group at the head lr.
    pcol_mod = None
    pcol_steps = pcol_cells = 0
    pcol_loss_sum = 0.0
    if args.pcol_w > 0:
        from loss_pcol import PCOL, grade_counts_from_labels
        _pc_counts = grade_counts_from_labels(labels, tr, prob_table,
                                              len(FINDINGS))
        pcol_mod = PCOL(len(FINDINGS), int(model.head.shape[1]), prob_table,
                        tau=args.pcol_tau, sigma=args.pcol_sigma,
                        grade_counts=_pc_counts).to(device)
        groups.append({"params": list(pcol_mod.parameters()), "lr": args.lr,
                       "weight_decay": 0.0})
        max_lr.append(args.lr)
        print(f"[pcol] w={args.pcol_w} tau={args.pcol_tau} sigma={args.pcol_sigma} "
              f"dim={int(model.head.shape[1])} train graded-cell counts "
              f"(finding x g0..g3):\n{_pc_counts}\n[pcol] grade weights:\n"
              f"{np.round(pcol_mod.w.cpu().numpy(), 3)}", flush=True)
        model.pcol_on = True
    if args.opt == "ademamix":
        from opt_ademamix import AdEMAMix
        opt = AdEMAMix(groups, betas=(0.9, 0.999), beta3=args.ademamix_beta3,
                       alpha=args.ademamix_alpha, warmup_steps=0,
                       weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    # --wd-from: locate the BACKBONE param group by identity, not by index.
    # The index is not stable — --freeze can drop empty groups and --stage-taps
    # adds one, and silently decaying the wrong group would be invisible.
    # --llrd < 1.0 splits the backbone across ONE GROUP PER DEPTH BUCKET (six
    # for convnext_tiny, fourteen for a 12-layer ViT), so "the backbone group"
    # is a set, not an index. The first version of this took the single
    # largest-overlap group and the smoke caught it immediately: it flipped
    # group 3 only, 85 of 180 backbone tensors, leaving 95 still decaying and
    # every log line looking healthy. Take EVERY group with backbone overlap,
    # and assert the tensors are fully covered.
    _wd_bb_groups = []
    if args.wd_from >= 0:
        _bb_ids = {id(p) for _, p in bb_named}
        # The PLACEBO target is everything the treatment is not, minus the
        # nodecay group (already at wd 0, so flipping it is a no-op that would
        # corrupt the coverage assertion).
        _want = _bb_ids if args.wd_target == "backbone" else (
            {id(p) for p in rest})
        _n_want = len(bb_named) if args.wd_target == "backbone" else len(rest)
        _cov = 0
        for _gi, _g in enumerate(opt.param_groups):
            _hit = sum(1 for p in _g["params"] if id(p) in _want)
            if _hit:
                if _hit != len(_g["params"]):
                    raise SystemExit(
                        f"--wd-from: param group {_gi} MIXES {_hit} "
                        f"{args.wd_target} with {len(_g['params']) - _hit} "
                        f"other params; decay cannot be changed for "
                        f"{args.wd_target} alone")
                _wd_bb_groups.append(_gi)
                _cov += _hit
        if not _wd_bb_groups:
            raise SystemExit(f"--wd-from: no param group contains "
                             f"{args.wd_target} params")
        if _cov != _n_want:
            raise SystemExit(f"--wd-from: covered {_cov} of {_n_want} "
                             f"{args.wd_target} tensors — refusing to run a "
                             f"partial intervention")
        print(f"[wdecay] target={args.wd_target}; param groups "
              f"{_wd_bb_groups} covering {_cov}/{_n_want} tensors; "
              f"weight_decay {args.wd} -> {args.wd * args.wd_scale} "
              f"at ep{args.wd_from}", flush=True)
    steps = max(1, len(tr_dl) // args.accum) * args.epochs
    if args.opt == "ademamix":
        _amx_T = int(round(args.ademamix_warmup * steps))
        for _g in opt.param_groups:
            _g["warmup_steps"] = _amx_T
        print(f"[ademamix] alpha={args.ademamix_alpha} beta3={args.ademamix_beta3} "
              f"warmup T={_amx_T} of {steps} steps; schedule at "
              f"t=T/4,T/2,T: {[tuple(round(v, 5) for v in opt.schedule_at(int(_amx_T * f))) for f in (0.25, 0.5, 1.0)]}",
              flush=True)
    if getattr(args, "lr_sched", "onecycle") == "cosine":
        # scratch/truebatch: warmup + cosine. Group LRs are set to max_lr so
        # LambdaLR's base_lrs equal OneCycle's peak per group.
        for _g, _m in zip(opt.param_groups, max_lr):
            _g["lr"] = _m
        _W = max(1, int(round(args.warmup_pct * steps)))
        def _cos_lambda(t, _W=_W, _T=steps):
            if t < _W:
                return 0.04 + 0.96 * t / _W
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (t - _W) / max(1, _T - _W))))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _cos_lambda)
        print(f"[lr-sched] cosine: warmup {_W}/{steps} steps from max_lr/25, "
              f"peak {['%.2e' % m for m in max_lr]}", flush=True)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, total_steps=steps, pct_start=args.warmup_pct)
    if args.warmup_pct != 0.1:
        print(f"[warmup] OneCycle pct_start {args.warmup_pct} = "
              f"{int(round(args.warmup_pct * steps))} of {steps} steps "
              f"(production 0.1)", flush=True)
    if args.sched_advance > 0:
        # Resume the SOURCE run's schedule at the position its checkpoint was
        # taken from, so the only difference between this arm and its own
        # control is the freeze. The loop below still runs --epochs epochs;
        # its `sched.last_epoch < steps - 1` guard parks the LR at the
        # schedule floor (max_lr/25/1e4, i.e. ~0) for the overhanging tail,
        # so the extra fraction of an epoch is a no-op rather than a restart.
        _adv = int(round(args.sched_advance * (steps / args.epochs)))
        if _adv >= steps:
            raise SystemExit(f"--sched-advance {args.sched_advance} = {_adv} "
                             f"steps >= the whole schedule ({steps})")
        for _ in range(_adv):
            sched.step()
        _lrs = ["%.3e" % g["lr"] for g in opt.param_groups]
        print(f"[sched-advance] OneCycle resumed at epoch "
              f"{args.sched_advance} = step {_adv}/{steps}; lr now {_lrs}",
              flush=True)
    # --lr-tail-frac (scratch/consttail/PREREG.md): the schedule stops being
    # stepped once sched.last_epoch reaches _tail_hold, so the LR stays at the
    # OneCycle value of that step for the rest of training. With the flag at
    # 0, _tail_hold == steps and min(steps - 1, _tail_hold) == steps - 1: the
    # guard at the step site is the pre-existing one, bit-identical.
    _tail_hold = steps
    _tail_announced = False
    if args.lr_tail_frac > 0:
        if not 0.0 < args.lr_tail_frac < 0.9:
            raise SystemExit(f"--lr-tail-frac {args.lr_tail_frac} must be in "
                             f"(0, 0.9)")
        _tail_hold = int(round((1.0 - args.lr_tail_frac) * steps))
        print(f"[lr-tail] OneCycle anneal stops at step {_tail_hold}/{steps} "
              f"(frac {args.lr_tail_frac} = last {steps - _tail_hold} steps "
              f"= {args.lr_tail_frac * args.epochs:.2f} epochs held); EMA "
              f"unchanged", flush=True)
    # --const-lr (scratch/forktest/PREREG.md): a continuation fork must hold a
    # FIXED LR, because an arm-vs-arm comparison at matched step count is only
    # clean if every arm sees the same LR at every step. The OneCycle object is
    # still built (so nothing below it needs a None-guard) but is never stepped.
    _const_lr = args.const_lr > 0
    if _const_lr:
        _mx = max(max_lr)
        for _g, _m in zip(opt.param_groups, max_lr):
            _g["lr"] = args.const_lr * (_m / _mx)
        print(f"[const-lr] OneCycle DISABLED; {len(opt.param_groups)} group lr "
              f"{['%.3e' % g['lr'] for g in opt.param_groups]} "
              f"(base {args.const_lr:.3e}, ratios from max_lr)", flush=True)
    _opt_steps = 0
    _steps_done = False
    if args.max_steps >= 0:
        print(f"[max-steps] training stops after {args.max_steps} optimizer "
              f"steps ({steps} would be the full schedule)", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    # ---- stage-tap probes (depth axis) -----------------------------------
    # Hooks pool IMMEDIATELY (a raw stage-1 map for 144 slices is ~0.5 GB;
    # GAP-pooled it is a few KB) and detach, so the main graph, the main
    # optimizer, the EMA and the checkpoint are all untouched.
    stage_taps, tap_feats, tap_probes, tap_opt = [], {}, None, None
    if args.stage_taps:
        _stages = getattr(model.backbone, "stages", None)
        if _stages is None:
            raise SystemExit(f"--stage-taps needs a backbone with .stages; "
                             f"{type(model.backbone).__name__} has none")
        stage_taps = sorted(int(x) for x in args.stage_taps.split(","))
        if stage_taps[-1] >= len(_stages) or stage_taps[0] < 0:
            raise SystemExit(f"--stage-taps {args.stage_taps!r} out of range "
                             f"for {len(_stages)} stages")

        def _mk_hook(si):
            def _h(m, i, o):
                tap_feats[si] = torch.nn.functional.adaptive_avg_pool2d(
                    o.detach().float(), 1).flatten(1)      # (N, C), detached
            return _h
        for si in stage_taps:
            _stages[si].register_forward_hook(_mk_hook(si))
        with torch.no_grad():
            model(torch.zeros(1, 1, args.n_slices, 64, 64, device=device),
                  torch.zeros(1, 1, dtype=torch.long, device=device),
                  torch.ones(1, 1, device=device),
                  torch.zeros(1, 1, dtype=torch.long, device=device))
        tap_dims = {si: tap_feats[si].shape[1] for si in stage_taps}
        tap_feats.clear()
        tap_probes = torch.nn.ModuleDict({
            f"s{si}": torch.nn.Sequential(
                torch.nn.LayerNorm(tap_dims[si]),
                torch.nn.Linear(tap_dims[si], len(FINDINGS)))
            for si in stage_taps}).to(device)
        tap_opt = torch.optim.AdamW(tap_probes.parameters(),
                                    lr=args.stage_tap_lr, weight_decay=0.0)
        print(f"[stage-taps] probes at stages {stage_taps}, dims {tap_dims}; "
              f"detached, separate AdamW lr {args.stage_tap_lr}; saved to "
              f"fold{args.fold}_stage_probes.pt", flush=True)

    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    # scratch/slicewit: the ev_head tap is registered AFTER the EMA shadow is
    # built (so the shadow never carries the hook) and armed ONLY around the
    # main training forward. Off -> None and nothing below touches the model.
    _sw_tap = EvHeadTap(model.ev_head) if slicewit is not None else None
    if ema is not None:
        eff = ema.effective_decay(steps)
        print(f"[ema] decay {args.ema_decay} (ramped) -> effective {eff:.6f} at "
              f"{steps} steps, window ~{1/(1-eff):.0f} steps; scored separately",
              flush=True)

    # Parallel shadows at other decays. EMA won 8/8 members against the raw
    # checkpoint, but the WINDOW was never swept - the shipped runs all sit at
    # the ramp's ceiling (~580 steps, 1.3 epochs) because every decay we have
    # ever passed is above the ramp. Shorter windows are the untested half.
    ema_extra = []
    if ema is not None and args.ema_extra_decays.strip():
        # Commas, semicolons or spaces: a comma-separated value cannot survive
        # sbatch --export=ALL,VAR=a,b (the shell hands SLURM two items), so the
        # launcher passes "0.995 0.998" and this accepts either spelling.
        for tok in args.ema_extra_decays.replace(";", " ").replace(",", " ").split():
            tok = tok.strip()
            if not tok:
                continue
            # "decay" or "decay@ramp": the ramp override is what makes decays
            # ABOVE the incumbent cap reachable at all (see EMA docstring).
            ramp = 10.0
            if "@" in tok:
                tok, _r = tok.split("@", 1)
                ramp = float(_r)
                if ramp < 1.0:
                    raise SystemExit(f"--ema-extra-decays: ramp {ramp} < 1")
            d = float(tok)
            if d <= 0 or d >= 1:
                raise SystemExit(f"--ema-extra-decays: {d} is not in (0,1)")
            tag = tok.replace(".", "") + ("" if ramp == 10.0 else f"r{ramp:g}")
            e2 = EMA(model, d, ramp=ramp)
            eff2 = e2.effective_decay(steps)
            if abs(eff2 - eff) < 1e-9:
                print(f"[ema] WARNING extra decay {d} ramps to the SAME "
                      f"effective {eff2:.6f} as the primary - this shadow is a "
                      f"duplicate, not a sweep point", flush=True)
            print(f"[ema] extra decay {d} -> effective {eff2:.6f}, window "
                  f"~{1/(1-eff2):.0f} steps -> fold{args.fold}_ema_d{tag}.pt",
                  flush=True)
            ema_extra.append({"decay": d, "tag": tag, "ema": e2, "best": -1.0})
        print(f"[ema] {len(ema_extra)} extra shadow(s), scored from ep"
              f"{args.ema_extra_start}", flush=True)

    # --mute-after: parsed once, up front, so a typo fails before the GPU time
    # is spent rather than being silently ignored for twelve epochs.
    mute = {}
    for spec in filter(None, (s.strip() for s in args.mute_after.split(","))):
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(f"--mute-after: expected FINDING:EPOCH[:WEIGHT], got {spec!r}")
        f = parts[0].strip()
        if f not in FINDINGS:
            raise SystemExit(f"--mute-after: {f!r} is not one of {FINDINGS}")
        mute[FINDINGS.index(f)] = (int(parts[1]), float(parts[2]) if len(parts) == 3 else 0.0)
    # --loss-cols (scratch/forktest/PREREG.md): the target-only loss mask. It
    # reuses `mute` rather than adding a second masking path, so every consumer
    # that already respects lw (soft_bce, the aux per-series head, --rank-w,
    # the pcol/text terms) is masked identically and no term can silently keep
    # teaching a muted column.
    if args.loss_cols.strip():
        if mute:
            raise SystemExit("--loss-cols and --mute-after are mutually "
                             "exclusive (both write the same weight vector)")
        _keepf = [c.strip() for c in args.loss_cols.split(",") if c.strip()]
        _bad = [c for c in _keepf if c not in FINDINGS]
        if _bad:
            raise SystemExit(f"--loss-cols: {_bad} not in {FINDINGS}")
        for _i, _f in enumerate(FINDINGS):
            if _f not in _keepf:
                mute[_i] = (0, 0.0)
        print(f"[loss-cols] keeping {_keepf}; {len(mute)} of "
              f"{len(FINDINGS)} columns weighted 0 from epoch 0", flush=True)
    if mute:
        print("[mute] " + ", ".join(
            f"{FINDINGS[i]} -> weight {w} from epoch {e}"
            for i, (e, w) in sorted(mute.items())), flush=True)

    # ---- --hard-neg FINDING:GATE:MULT -----------------------------------
    # The normaliser is computed over the TRAIN studies only, from the same
    # soft targets the loss sees, so the column's expected weight is exactly
    # preserved. Computing it over `studies` instead would leak the val split's
    # prevalence into the training objective.
    hard_neg = []
    for spec in filter(None, (s.strip() for s in args.hard_neg.split(","))):
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(
                f"--hard-neg: expected FINDING:GATE:MULT, got {spec!r}")
        f, g = parts[0].strip(), parts[1].strip()
        for nm in (f, g):
            if nm not in FINDINGS:
                raise SystemExit(f"--hard-neg: {nm!r} is not one of {FINDINGS}")
        mult = float(parts[2])
        gi = FINDINGS.index(g)
        gate = [1.0 if labels[u][gi] >= 0.5 else 0.0 for u in tr if u in labels]
        p = float(np.mean(gate)) if gate else 0.0
        nrm = p * mult + (1.0 - p)
        hard_neg.append((FINDINGS.index(f), gi, mult, nrm))
        print(f"[hard-neg] {f} x{mult} where {g} positive "
              f"({p:.1%} of {len(gate)} train studies), normaliser {nrm:.4f} "
              f"-> effective weight {mult/nrm:.3f} on gated / "
              f"{1/nrm:.3f} on ungated", flush=True)

    sel_cols = None
    if args.select_findings.strip():
        sel_cols = []
        for f in filter(None, (s.strip() for s in args.select_findings.split(","))):
            if f not in FINDINGS:
                raise SystemExit(f"--select-findings: {f!r} is not one of {FINDINGS}")
            sel_cols.append(FINDINGS.index(f))
        # A finding that selection scores on but training never sees is almost
        # certainly a typo in one of the two flags, and it would silently
        # select on an untrained column for the whole run.
        for j in sel_cols:
            if mute.get(j, (10 ** 9, 1.0))[0] == 0 and mute[j][1] == 0.0:
                raise SystemExit(
                    f"--select-findings {FINDINGS[j]!r} is muted to weight 0 "
                    f"from epoch 0 by --mute-after; selection would score a "
                    f"column that never receives gradient")
        print(f"[select] checkpoint selection scores only "
              f"{[FINDINGS[j] for j in sel_cols]}", flush=True)

    if args.rank_w > 0 and args.bs != 1:
        raise SystemExit(f"--rank-w needs --bs 1 (got {args.bs}): the rank block "
                         "indexes logits[0] and y[0], so rows 1..B-1 receive no "
                         "ranking gradient while all B are pushed to the bank")

    # FIFO of recent (logit, target) pairs for the ranking loss. bs=1 means a
    # forward pass sees ONE study, so within-batch pairing is impossible; the
    # bank supplies the opposite-class partners. Detached, so this adds no
    # backward cost and cannot leak gradient between studies.
    rank_bank = []

    # --bagauc: grade-bag pairwise AUC (scratch/bagauc/PREREG.md). Unlike
    # --rank-w's bank, partners here are NEVER detached - both logits in a
    # pair stay live in the SAME accumulation window and get ONE combined
    # backward at the window boundary, so the update genuinely CAN reorder
    # two studies (verified in scratch/bagauc/test_loss.py before any launch).
    if args.bagauc_w > 0 and args.bs != 1:
        raise SystemExit(f"--bagauc-w needs --bs 1 (got {args.bs}): the window "
                         "buffer indexes b['study'] and logits[0] per micro-step "
                         "assuming one study per forward")
    if args.bagauc_w > 0 and args.sam_rho > 0:
        raise SystemExit("--bagauc-w and --sam-rho are not implemented "
                         "together: SAM's ascent forward overwrites `logits` "
                         "before the retained graph this loss needs would be "
                         "consumed")
    bagauc_grade = {}
    bagauc_pi = None
    if args.bagauc_w > 0:
        with open(args.labels, encoding="utf-8") as _f:
            for _line in _f:
                _r = json.loads(_line)
                if "grades" not in _r:
                    continue   # probs-only (observer-v5) rows carry no bag
                bagauc_grade[_r["uid"]] = np.array(
                    [int(_r["grades"].get(k, -1)) for k in FINDINGS], dtype=np.int64)
        # Per-finding pi(bag), bag index 0=silent(-1), 1..4=grade 0..3 - the
        # EXACT table grade_to_prob used to build the soft targets `y`, so a
        # bag's "distance" is the same prior gap the soft-BCE already trusted.
        _pi_silent = (np.asarray(silent, dtype=np.float32) if silent is not None
                      else np.full(len(FINDINGS), prob_table[-1], np.float32))
        bagauc_pi = np.stack([
            _pi_silent,
            np.full(len(FINDINGS), prob_table[0], np.float32),
            np.full(len(FINDINGS), prob_table[1], np.float32),
            np.full(len(FINDINGS), prob_table[2], np.float32),
            np.full(len(FINDINGS), prob_table[3], np.float32),
        ], axis=1)   # (12, 5)
        bagauc_pi_t = torch.from_numpy(bagauc_pi).to(device)
        print(f"[bagauc] w={args.bagauc_w} start_ep={args.bagauc_start_ep} "
              f"margin_form={args.bagauc_margin_form}; grades resolved for "
              f"{len(bagauc_grade)} studies; per-finding pi(bag) "
              f"[silent,g0,g1,g2,g3]:", flush=True)
        for _j, _f in enumerate(FINDINGS):
            print(f"   {_f:<18} " + " ".join(f"{v:.3f}" for v in bagauc_pi[_j]),
                  flush=True)
    # --silent-mute-from: grade lookup for the silent mask. DELIBERATELY a
    # second dict rather than a refactor that shares bagauc_grade - the
    # default path must stay byte-for-byte untouched, which is the same rule
    # _bagauc_gc_local_loss follows a few hundred lines down. Populated only
    # when the flag is on, so an unflagged arm pays nothing (not even the
    # file read).
    silmute_grade = {}
    if args.silent_mute_from >= 0:
        if args.bagauc_w > 0:
            raise SystemExit(
                "--silent-mute-from and --bagauc-w are not implemented "
                "together. That combination is gsplit (graded BCE + silent "
                "bag-AUC), which scratch/bagauc/RESULTS.md 7 demotes: for "
                "SILENT cells bagauc and gsplit are identical, and bagauc_v1 "
                "already measured that treatment at Syno|silent 0.679. A "
                "guard is safer than an untested interaction.")
        with open(args.labels, encoding="utf-8") as _f:
            for _line in _f:
                _r = json.loads(_line)
                if "grades" not in _r:
                    continue   # probs-only rows: absent is NOT silent
                silmute_grade[_r["uid"]] = np.array(
                    [int(_r["grades"].get(k, -1)) for k in FINDINGS],
                    dtype=np.int64)
        if args.silent_mute_findings.strip():
            # scratch/silweight: restrict the mute to named findings by marking
            # every other column -99 (= "absent, never silent") so the
            # downstream `== -1` test leaves them untouched.
            _keep = [f.strip() for f in args.silent_mute_findings.split(",") if f.strip()]
            _bad = [f for f in _keep if f not in FINDINGS]
            if _bad:
                raise SystemExit(f"--silent-mute-findings unknown: {_bad}")
            _off = np.array([k not in _keep for k in FINDINGS])
            for _u in silmute_grade:
                silmute_grade[_u] = np.where(_off, -99, silmute_grade[_u])
            print(f"[silmute] restricted to findings {_keep}")
        _sm_counts = np.zeros(len(FINDINGS), dtype=np.int64)
        for _g in silmute_grade.values():
            _sm_counts += (_g == -1)
        _sm_n = max(len(silmute_grade), 1)
        print(f"[silmute] from ep{args.silent_mute_from} silent-cell BCE "
              f"weight x{args.silent_mute_w}; grades resolved for "
              f"{len(silmute_grade)} studies; muted cells "
              f"{int(_sm_counts.sum())} of {_sm_n * len(FINDINGS)} "
              f"({100.0 * _sm_counts.sum() / (_sm_n * len(FINDINGS)):.1f}%). "
              f"Per-finding silent share:", flush=True)
        for _j, _f in enumerate(FINDINGS):
            print(f"   {_f:<18} {int(_sm_counts[_j]):5d}  "
                  f"{100.0 * _sm_counts[_j] / _sm_n:5.1f}%", flush=True)
    # --report-cell-weight (scratch/rfsyno): per-finding weight applied to
    # REPORT rows only. A report row is one whose --labels record carries an
    # 'evidence' key (extraction output); cotrain/OAI rows have none. Verified
    # at load against the --cotrain-pool set (must be disjoint) so a silent
    # mis-identification cannot under- or over-dose the arm. Populated only
    # when the flag is set; unflagged arms never touch the weight tensor.
    rcw_report, rcw_col = set(), None
    if args.report_cell_weight.strip():
        _rc = {}
        for _tok in args.report_cell_weight.split(","):
            if not _tok.strip():
                continue
            _f, _v = _tok.rsplit(":", 1)
            if _f.strip() not in FINDINGS:
                raise SystemExit(f"--report-cell-weight unknown finding {_f!r}")
            _rc[_f.strip()] = float(_v)
        rcw_col = np.array([_rc.get(k, 1.0) for k in FINDINGS], np.float32)
        _n_rows = 0
        with open(args.labels, encoding="utf-8") as _f:
            for _line in _f:
                _r = json.loads(_line)
                _n_rows += 1
                if "grades" in _r and _r.get("evidence") is not None:
                    rcw_report.add(_r["uid"])
        _ovl = len(rcw_report & cotrain)
        if _ovl:
            raise SystemExit(f"--report-cell-weight: {_ovl} report rows are "
                             "also in --cotrain-pool; identification invalid")
        _zero = [k for k in FINDINGS if _rc.get(k, 1.0) == 0.0]
        print(f"[rcw] --report-cell-weight {_rc}; report rows {len(rcw_report)} "
              f"of {_n_rows} label rows (cotrain rows {len(cotrain)}, overlap "
              f"{_ovl}); zero-weighted cells {len(rcw_report) * len(_zero)} "
              f"({_zero}) from ep0", flush=True)
        rcw_col_t = torch.from_numpy(rcw_col).to(device)
    bagauc_buf = []              # live (logits, grade, w) tuples this window
    bagauc_pair_counts = np.zeros(len(FINDINGS), dtype=np.int64)
    bagauc_n_windows = 0
    # --bagauc-gradcache: raw CPU batches for the CURRENT window, buffered
    # instead of live logits (bagauc_buf above stays empty and its two
    # consumers below become no-ops) - never populated unless
    # args.bagauc_gradcache, so a plain --bagauc-w run never allocates this.
    bagauc_gc_buf = []

    def _bagauc_window_loss(buf):
        """Cross-bag pairwise logistic loss over one accumulation window.

        `buf` is a list of (logits, grade, weight) triples, one per
        micro-step, ALL still attached to their own live graphs (never
        detached) - concatenated here so a pair can span any two micro-steps
        of the window. Per finding: valid cells are those with a known bag
        and nonzero weight; PI = pi_row - pi_col is the constant bag-distance
        (no grad); only PI>0 entries contribute, at weight PI, via
        softplus(-(z_row-z_col)) - which is exactly zero whenever every valid
        cell shares one bag (PI all zero) and nonzero-gradient-to-BOTH-sides
        whenever a higher-bag study's logit sits below a lower-bag study's.
        Returns (loss or None, pair_counts (12,) int64); None means no
        finding in this window had a cross-bag pair (fully degenerate window).
        """
        logits_all = torch.cat([t[0] for t in buf], dim=0).float()   # (N,12)
        grade_all = torch.cat([t[1] for t in buf], dim=0)            # (N,12)
        w_all = torch.cat([t[2] for t in buf], dim=0)                # (N,12)
        pair_counts = np.zeros(len(FINDINGS), dtype=np.int64)
        col_losses = []
        for j in range(len(FINDINGS)):
            valid = (grade_all[:, j] != -99) & (w_all[:, j] > 0)
            idx = valid.nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            z = logits_all[idx, j]                      # (n,) LIVE
            g = grade_all[idx, j]                        # (n,) bag id -1..3
            pi = bagauc_pi_t[j][g + 1]                    # (n,) constant, no grad
            col_loss, n_pairs = bagauc_pair_loss(z, pi)
            if col_loss is None:
                continue                                  # all one bag: 0
            pair_counts[j] = n_pairs
            col_losses.append(col_loss)
        if not col_losses:
            return None, pair_counts
        # Fixed-12-column mean, same convention soft_bce/--mute-after use: a
        # finding with no pair this window contributes 0, not a rescale of
        # the findings that did get pairs.
        return torch.stack(col_losses).sum() / len(FINDINGS), pair_counts

    # --pair-rank: longitudinal monotonicity on OA. Both scans of a pair are
    # forwarded JOINTLY in the same micro-step, so the penalty reorders both -
    # unlike the bank-based ranking loss above, whose fixed partner made the
    # update a monotone map that could never change AUC. The 96m partner has
    # no labels and is reached only here.
    pair_iter, pair_oa = None, [FINDINGS.index(f)
                                for f in ("Medial OA", "Lateral OA", "PF OA")]
    if args.pair_rank > 0:
        pmap = json.load(open(args.pair_map))
        have_series = set(series_index)
        # Two formats. dict {a: b} = the original longitudinal map: b is the
        # LATER scan and the penalised columns are the monotone-progression OA
        # trio (direction is free, degeneration does not reverse). list of
        # [a, b, col] = DIRECTED pairs: b must outrank a on exactly `col`,
        # direction supplied by measured grade deltas (synovitis fluctuates,
        # so its direction cannot be assumed from time order).
        if isinstance(pmap, dict):
            cand = [(a, b, pair_oa) for a, b in sorted(pmap.items())]
        else:
            cand = [(a, b, [int(c)]) for a, b, c in pmap]
        usable = [(a, b, cols) for a, b, cols in cand
                  if a in have_series and b in have_series
                  and os.path.isdir(os.path.join(args.vol_root, a))
                  and os.path.isdir(os.path.join(args.vol_root, b))]
        if not usable:
            raise SystemExit(f"--pair-rank: no usable pairs from {args.pair_map} "
                             f"under {args.vol_root}")
        pair_uids = sorted({u for a, b, _ in usable for u in (a, b)})
        pair_base = KneeStudyDataset(
            pair_uids, series_index, None, args.vol_root, args.n_series,
            args.n_slices, args.size, True, laterality=lat, geometry=geom,
            sequence=seqmap, pfrel_series=(_pfrel_mode != "off"), bone_series=(_bone_mode != "off"), bone_fluid=(0 if args.bone_series_fluid == "nonfs" else 1), native_ctx=args.native_ctx, slice_ctx=args.slice_ctx, ctx_mm=args.ctx_mm, zspacing=(load_zspacing() if args.ctx_mm > 0 else None), aspect=(load_aspect() if args.aspect_fix else None), full_fov=args.full_fov, xfuse_pairs=_xf_pairs)
        pidx = {u: k for k, u in enumerate(pair_uids)}

        class _PairDS(torch.utils.data.Dataset):
            def __len__(self):
                return len(usable)

            def __getitem__(self, k):
                a, b, cols = usable[k]
                ia, ib = pair_base[pidx[a]], pair_base[pidx[b]]
                return {"a_vol": ia["vol"], "a_st": ia["stype"],
                        "a_seq": ia["seq"], "a_mask": ia["mask"],
                        "b_vol": ib["vol"], "b_st": ib["stype"],
                        "b_seq": ib["seq"], "b_mask": ib["mask"],
                        "cols": torch.tensor(cols, dtype=torch.long)}

        pair_dl = DataLoader(_PairDS(), batch_size=1, shuffle=True,
                             num_workers=2, pin_memory=True,
                             persistent_workers=True)

        def _cycle(dl):
            # NOT itertools.cycle: that caches every yielded item, and one
            # epoch of pairs is ~250 GB of volume tensors.
            while True:
                for x in dl:
                    yield x

        pair_iter = _cycle(pair_dl)
        _colset = sorted({c for _, _, cs in usable for c in cs})
        print(f"[pair] {len(usable)} longitudinal pairs, weight "
              f"{args.pair_rank}, every {args.pair_every} steps, columns "
              f"{[FINDINGS[j] for j in _colset]}", flush=True)

    # --bilat-rank: BILATERAL same-subject same-visit L/R ranking on the OA
    # trio (scratch/oaibilat/PREREG.md). Same joint-forward structure as
    # --pair-rank above, but the pair is cross-sectional (left vs right knee
    # of one person in one session, so person/site/scanner nuisance cancels
    # exactly) and each compartment carries its OWN direction. Discordance
    # and strata (STRONG >= 2 ordinal steps / MILD 1 / TIE excluded) are
    # frozen at build time from the depth-aware rubric severities
    # (scratch/oaibilat/build_strata.py via the frozen latoarescue sidecar).
    # Flag-gated and ADDITIVE: with the default 0 nothing below is
    # constructed and the loop guard (`bilat_iter is not None`) never fires.
    bilat_iter = None
    bilat_tie_comps = 0
    if args.bilat_rank > 0:
        if args.pair_rank > 0:
            raise SystemExit("--bilat-rank + --pair-rank is an untested "
                             "combination (two independent pair iterators "
                             "on the OA columns); not authorized")
        if args.bagauc_w > 0 or args.sam_rho > 0:
            raise SystemExit("--bilat-rank is not implemented for the "
                             "--bagauc-w / --sam-rho loss paths (each owns "
                             "the loss/step assembly and neither is tested "
                             "with this term)")
        _bj = json.load(open(args.bilat_pairs))
        _b_src = len(_bj["pairs"])
        _b_have = set(series_index)
        usable_b, _b_tieonly, _b_nomedia, _b_notpool = [], 0, 0, 0
        _b_terms = {s: 0 for s in BILAT_STRATA}
        for _pr in _bj["pairs"]:
            _a, _b = _pr["left"], _pr["right"]
            _cols, _dirs, _strs = [], [], []
            for _cn in sorted(_pr["comp"]):
                _rc = _pr["comp"][_cn]
                if _rc["stratum"] == "TIE":
                    bilat_tie_comps += 1
                    continue
                if _rc["stratum"] == "MASKED":
                    continue
                _cols.append(FINDINGS.index(_rc["finding"]))
                _dirs.append(1.0 if _rc["worse"] == "L" else -1.0)
                _strs.append(BILAT_STRATA.index(_rc["stratum"]))
            if not _cols:
                _b_tieonly += 1          # tie/masked-only pair: excluded
                continue
            if not (_a in _b_have and _b in _b_have
                    and os.path.isdir(os.path.join(args.vol_root, _a))
                    and os.path.isdir(os.path.join(args.vol_root, _b))):
                _b_nomedia += 1
                continue
            if cotrain and not (_a in cotrain and _b in cotrain):
                _b_notpool += 1          # OAI rows only ride the pool
                continue
            for _s, _si in zip(BILAT_STRATA, range(len(BILAT_STRATA))):
                _b_terms[_s] += sum(1 for x in _strs if x == _si)
            usable_b.append((_a, _b, _cols, _dirs, _strs))
        if not usable_b:
            raise SystemExit(f"--bilat-rank: no usable discordant pairs "
                             f"from {args.bilat_pairs} under "
                             f"{args.vol_root}")
        bilat_uids = sorted({u for _a, _b, *_ in usable_b for u in (_a, _b)})
        bilat_base = KneeStudyDataset(
            bilat_uids, series_index, None, args.vol_root, args.n_series,
            args.n_slices, args.size, True, laterality=lat, geometry=geom,
            sequence=seqmap, pfrel_series=(_pfrel_mode != "off"), bone_series=(_bone_mode != "off"), bone_fluid=(0 if args.bone_series_fluid == "nonfs" else 1), native_ctx=args.native_ctx, slice_ctx=args.slice_ctx, ctx_mm=args.ctx_mm, zspacing=(load_zspacing() if args.ctx_mm > 0 else None), aspect=(load_aspect() if args.aspect_fix else None), full_fov=args.full_fov, xfuse_pairs=_xf_pairs)
        _bidx = {u: k for k, u in enumerate(bilat_uids)}

        class _BilatDS(torch.utils.data.Dataset):
            def __len__(self):
                return len(usable_b)

            def __getitem__(self, k):
                _a, _b, _cols, _dirs, _strs = usable_b[k]
                ia, ib = bilat_base[_bidx[_a]], bilat_base[_bidx[_b]]
                return {"l_vol": ia["vol"], "l_st": ia["stype"],
                        "l_seq": ia["seq"], "l_mask": ia["mask"],
                        "r_vol": ib["vol"], "r_st": ib["stype"],
                        "r_seq": ib["seq"], "r_mask": ib["mask"],
                        "cols": torch.tensor(_cols, dtype=torch.long),
                        "dirs": torch.tensor(_dirs, dtype=torch.float32),
                        "strata": torch.tensor(_strs, dtype=torch.long)}

        bilat_dl = DataLoader(_BilatDS(), batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True)

        def _bilat_cycle(dl):
            # NOT itertools.cycle (caches every yielded volume tensor).
            while True:
                for x in dl:
                    yield x

        bilat_iter = _bilat_cycle(bilat_dl)
        print(f"[bilat] {len(usable_b)} discordant L/R pairs of {_b_src} "
              f"built (excluded: tie/masked-only {_b_tieonly}, no media "
              f"{_b_nomedia}, outside cotrain pool {_b_notpool}); "
              f"comp-terms STRONG {_b_terms['STRONG']} MILD "
              f"{_b_terms['MILD']} (TIE excluded at build: "
              f"{bilat_tie_comps}); w {args.bilat_rank}, every "
              f"{args.bilat_every} micro-steps", flush=True)

    # --aux-pool-*: step-matched auxiliary pool stream (scratch/noisystud/
    # PREREG_dose.md, Addendum D). The main loader `tr_dl` is not touched, so
    # steps/epoch, OneCycle total_steps, EMA and snapshot indices are the
    # control's. The aux loader has its OWN torch.Generator (seed + 7919):
    # RandomSampler and the worker base seed both draw from it, so neither
    # building nor iterating it consumes the global RNG that orders the main
    # loader. Off (empty --aux-pool-teacher): nothing below runs and the loop
    # guard (`auxpool_iter is not None`) never fires.
    _moh_state = [None, []]          # --main-order-hash: [digest, uids]
    auxpool_iter = None
    auxpool_draws = 0
    auxpool_mass = 0.0
    auxpool_main_mass = {"rsna": 0.0, "pool": 0.0}
    auxpool_loss_sum = 0.0
    if args.aux_pool_teacher:
        _ap_bad = [n for n, v in (
            ("--sam-rho", args.sam_rho > 0), ("--bagauc-w", args.bagauc_w > 0),
            ("--plane-filter", bool(args.plane_filter)),
            ("--xfuse", bool(args.xfuse)),
            ("--witness-targets", bool(args.witness_targets)),
            ("--slicewit-targets", bool(args.slicewit_targets)),
            ("--no-gold", bool(args.no_gold))) if v]
        if _ap_bad:
            raise SystemExit(f"--aux-pool-teacher is not implemented with "
                             f"{_ap_bad} (PREREG_dose.md D1 scope)")
        if not (args.aux_pool_vol_root and args.aux_pool_series_csv
                and args.aux_pool_laterality_map):
            raise SystemExit("--aux-pool-teacher requires --aux-pool-vol-root, "
                             "--aux-pool-series-csv and "
                             "--aux-pool-laterality-map")
        if args.aux_pool_every < 1 or not (args.aux_pool_w > 0):
            raise SystemExit("--aux-pool-every must be >= 1 and --aux-pool-w "
                             "> 0 when --aux-pool-teacher is set")
        # Same validation as --teacher-targets.
        _az = np.load(args.aux_pool_teacher, allow_pickle=True)
        if "uids" not in _az or "targets" not in _az or "findings" not in _az:
            raise SystemExit("--aux-pool-teacher must contain uids, targets, "
                             "findings")
        if list(str(x) for x in _az["findings"]) != FINDINGS:
            raise SystemExit("--aux-pool-teacher finding order differs from "
                             "FINDINGS")
        _au = [str(x) for x in _az["uids"]]
        _ap = np.asarray(_az["targets"], dtype=np.float32)
        if _ap.shape != (len(_au), len(FINDINGS)):
            raise SystemExit(f"--aux-pool-teacher shape {_ap.shape} does not "
                             f"match ({len(_au)}, {len(FINDINGS)})")
        if len(set(_au)) != len(_au):
            raise SystemExit("--aux-pool-teacher contains duplicate UIDs")
        if not np.isfinite(_ap).all() or _ap.min() < 0.0 or _ap.max() > 1.0:
            raise SystemExit("--aux-pool-teacher contains non-finite or "
                             "out-of-range probabilities")
        # Disjointness: gold-58, every main-pool study (RSNA + OAI, i.e. the
        # main series index), the train split and the val fold.
        _ap_main = set(series_index) | set(tr) | set(va) | set(gold)
        _ap_ovl = sorted(_ap_main.intersection(_au))
        if _ap_ovl:
            raise SystemExit(f"--aux-pool-teacher overlaps RSNA|OAI|val|gold "
                             f"on {len(_ap_ovl)} uids (e.g. {_ap_ovl[:3]}); "
                             f"refusing")
        _ap_si = load_series_index(args.aux_pool_series_csv)
        _ap_lat = load_laterality(args.aux_pool_laterality_map)
        _ap_use = [u for u in _au if u in _ap_si and os.path.isdir(
            os.path.join(args.aux_pool_vol_root, u))]
        if len(_ap_use) != len(_au):
            raise SystemExit(f"--aux-pool-teacher: {len(_au) - len(_ap_use)} "
                             f"of {len(_au)} uids have no series/volume under "
                             f"{args.aux_pool_vol_root}; refusing a silently "
                             f"smaller pool")
        _ap_lab = {u: row for u, row in zip(_au, _ap)}
        # Training-mode transform with the main dataset's noise settings;
        # labels = teacher probabilities, no coverage (all-ones), no label
        # repair / evidence / pairmax / aux labels.
        auxpool_ds = KneeStudyDataset(
            _ap_use, _ap_si, _ap_lab, args.aux_pool_vol_root, args.n_series,
            args.n_slices, args.size, True, laterality=_ap_lat, geometry=geom,
            sequence=seqmap,
            flip_swap=args.flip_swap, cutmix=args.cutmix,
            cutmix_alpha=args.cutmix_alpha, coverage=None,
            domain_rand=args.domain_rand, vendor_aug=args.vendor_aug,
            series_dropout=args.series_dropout, spatial_aug=args.spatial_aug,
            affine_p=args.affine_p,
            pfrel_series=(_pfrel_mode != "off"),
            bone_series=(_bone_mode != "off"), bone_fluid=(0 if args.bone_series_fluid == "nonfs" else 1), native_ctx=args.native_ctx,
            slice_ctx=args.slice_ctx, ctx_mm=args.ctx_mm,
            zspacing=(load_zspacing() if args.ctx_mm > 0 else None),
            aspect=(load_aspect() if args.aspect_fix else None),
            full_fov=args.full_fov)
        _ap_seed = ((args.seed + 7919) if args.seed >= 0
                    else int.from_bytes(os.urandom(4), "little"))
        _ap_gen = torch.Generator()
        _ap_gen.manual_seed(_ap_seed)
        auxpool_dl = DataLoader(auxpool_ds, batch_size=1, shuffle=True,
                                num_workers=2, pin_memory=True,
                                persistent_workers=True, generator=_ap_gen)

        def _auxpool_cycle(dl):
            # NOT itertools.cycle (caches every yielded volume tensor).
            while True:
                for x in dl:
                    yield x

        auxpool_iter = _auxpool_cycle(auxpool_dl)
        _ap_spe = max(1, len(tr_dl) // args.accum)
        print(f"[auxpool] N={len(_ap_use)} every={args.aux_pool_every} "
              f"w={args.aux_pool_w:g} overlap(RSNA|OAI|val|gold)=0 main "
              f"steps/epoch={_ap_spe} total={_ap_spe * args.epochs} "
              f"(generator seed {_ap_seed}, source {args.aux_pool_teacher})",
              flush=True)

    # --dann: gradient-reversal site adversary. The head is a separate module
    # with its own optimizer so the model's state_dict (and every downstream
    # loader) is untouched. Domain = manufacturer, normalised to 4 classes.
    dann_head = dann_opt = None
    dom_of = {}
    if args.dann > 0 and args.site_aux > 0:
        raise SystemExit("--dann and --site-aux share one head and pull the "
                         "trunk in opposite directions; pick one")
    if args.dann > 0 or args.site_aux > 0:
        from site_shift_probe import load_site_meta

        def _dom(m):
            m = (m or "").upper()
            if m.startswith("GE"):
                return 0
            if "PHILIPS" in m:
                return 1
            if "SIEMENS" in m:
                return 2
            return 3

        dom_of = {u: _dom(v.get("manufacturer"))
                  for u, v in load_site_meta().items()}

        class _GRL(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lam):
                ctx.lam = lam
                return x.view_as(x)

            @staticmethod
            def backward(ctx, g):
                return -ctx.lam * g, None

        dann_head = torch.nn.Sequential(
            torch.nn.Linear(model.dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 4)).to(device)
        dann_opt = torch.optim.AdamW(dann_head.parameters(), lr=1e-4)
        _grl = _GRL.apply
        from collections import Counter as _Counter
        cov = _Counter(dom_of.get(u, 3) for u in tr)
        _mode = "dann (reversed)" if args.dann > 0 else f"site-aux {args.site_aux} (plain)"
        print(f"[site-head] {_mode}; train domain counts "
              f"GE/Philips/Siemens/other = "
              f"{[cov.get(k, 0) for k in range(4)]}", flush=True)

    # --aucm: AUC-margin saddle-point surrogate (LibAUC AUCMLoss adapted to
    # soft ordinal targets). All state is external to the model so every
    # checkpoint loader downstream is untouched. alpha is the DUAL variable
    # and is maximised, so it gets a manual ascent step at the accumulation
    # boundary rather than a seat in an optimizer that minimises.
    aucm_idx, aucm_opt = [], None
    if args.aucm > 0:
        for f in filter(None, (s.strip() for s in args.aucm_findings.split(","))):
            if f not in FINDINGS:
                raise SystemExit(f"--aucm-findings: {f!r} not in {FINDINGS}")
            aucm_idx.append(FINDINGS.index(f))
        if not aucm_idx:
            raise SystemExit("--aucm > 0 but --aucm-findings is empty")
        _J = len(aucm_idx)
        aucm_a = torch.zeros(_J, device=device, requires_grad=True)
        aucm_b = torch.zeros(_J, device=device, requires_grad=True)
        aucm_opt = torch.optim.SGD([aucm_a, aucm_b], lr=args.aucm_lr)
        aucm_alpha = torch.ones(_J, device=device)
        aucm_p = torch.full((_J,), 0.3, device=device)   # EMA soft-pos mass
        aucm_agrad = torch.zeros(_J, device=device)
        print(f"[aucm] w={args.aucm} margin={args.aucm_margin} "
              f"lr={args.aucm_lr} on {[FINDINGS[j] for j in aucm_idx]}; "
              f"the other {12 - _J} findings are the placebo", flush=True)

    os.makedirs(args.out, exist_ok=True)

    # Sub-epoch snapshot schedule. Parsed once, up front, so a malformed
    # fraction fails at startup rather than five hours into a run.
    _sub_fracs = [float(x) for x in args.save_subepoch.split(",") if x.strip()]
    assert all(0.0 < f < 1.0 for f in _sub_fracs), \
        f"--save-subepoch fractions must be strictly inside (0,1): {_sub_fracs}"
    _sub_eps = set(int(x) for x in args.save_subepoch_epochs.split(",") if x.strip())
    if _sub_fracs:
        print(f"[subepoch] EMA snapshots at fractions {_sub_fracs} of epochs "
              f"{sorted(_sub_eps)}", flush=True)

    best, best_ema, best_ema_a5 = -1.0, -1.0, -1.0
    sam_state = sam_params = None
    if args.sam_rho > 0:
        # SAM's second forward recomputes exactly soft_bce + aux
        # (+ gold_train_w). Any loss term outside that set would silently
        # vanish from the SAM gradient, so refuse the combination outright.
        assert (not comp_pairs and args.rank_w == 0 and dann_head is None
                and not aucm_idx and pair_iter is None and tap_probes is None
                and not aux_names and not text_dim and not hard_neg
                and not pairmax and args.rep_gls_sharp == 0
                and args.pair_route_reg == 0 and cotrain_sched is None), \
            "--sam-rho supports the plain recipe only (soft_bce+aux+gold_w)"
        sam_params = [p for g in opt.param_groups for p in g["params"]
                      if p.requires_grad]
        sam_state = {"e": {}, "acc": {}}
        print(f"[sam] rho={args.sam_rho}  m=1 per-micro-batch, manual accum "
              f"({len(sam_params)} param tensors); step/EMA/snapshot indices "
              f"identical to a plain run; ~2x forward cost", flush=True)

    # Frozen-parameter integrity reference. requires_grad=False is not by
    # itself proof: a param handed to the optimizer can still be moved by
    # decoupled weight decay, and a stale .grad survives zero_grad(set_to_none)
    # only if something wrote it. Checked every epoch so the run dies at ep0
    # rather than producing a quietly-unfrozen result 10 epochs later.
    # Every frozen tensor, not a sample: the clones cost one extra copy of the
    # frozen weights (~113 MB for a frozen convnext_tiny trunk), far less than
    # the two Adam moment buffers the freeze already saved.
    _frz_ref = None
    if frozen_params:
        _frz_ref = [(n, p, p.detach().clone()) for n, p in frozen_params]
        print(f"[freeze] integrity reference cloned for ALL {len(_frz_ref)} "
              f"frozen tensors "
              f"({sum(r.numel() for _, _, r in _frz_ref)/1e6:.2f}M params)",
              flush=True)

    def _frz_check(where):
        if not _frz_ref:
            return
        for n, p, ref in _frz_ref:
            if not torch.equal(p.detach(), ref):
                raise SystemExit(f"[freeze] VIOLATION at {where}: frozen "
                                 f"parameter {n} changed "
                                 f"(max|d|={float((p.detach()-ref).abs().max()):.3e})")
        print(f"[freeze] integrity OK at {where}: all {len(_frz_ref)} frozen "
              f"tensors BIT-IDENTICAL to their load-time values", flush=True)

    # ---- --bagauc-gradcache: GradCache two-pass window processing --------
    # (scratch/bagauc/PREREG.md gradcache addendum, 2026-08-20). Defined
    # here, once, rather than inline in the epoch loop, but closes over
    # per-epoch locals (lw, mute) exactly like the plain path does - Python
    # resolves free variables at CALL time, and every call happens deep
    # inside the epoch loop, long after lw/dann_head/aucm_idx/etc. exist.
    #
    # _bagauc_gc_weight duplicates the weight/grade computation that the
    # plain per-microstep block does inline (lines ~2170-2230 below) so it
    # can be called TWICE per study - once (no_grad) in pass A, once in pass
    # C - with identical, purely-deterministic results (none of it touches
    # the model). _bagauc_gc_local_loss duplicates the loss-assembly logic
    # of that same block (every optional term: pairmax, comp-pair, text,
    # aux, rank-w, report-aux, pair-route-reg, gold_train_w, dann, aucm) so
    # pass C reproduces EXACTLY what a non-gradcache micro-step would have
    # contributed, modulo only the deferred bagauc pairwise term. This is
    # deliberate duplication, not a refactor of the existing inline block:
    # the plain (non-gradcache) path must stay byte-for-byte untouched, so
    # nothing here may be shared code the default path also executes.
    # ---- --gram-escrow: anchor cache -------------------------------------
    # scratch/gramesc/PREREG.md (+ amendment 2). Built ONCE, at the fractional
    # epoch named by --gram-escrow-from, from the EMA weights - the state whose
    # Syno|silent actually measures 0.875-0.879 is the ep0.30 EMA snapshot, so
    # that is the thing worth preserving; the live raw model is what gets
    # constrained toward it.
    gram_href = None        # (K, D) normalised reference reps, CONSTANT
    gram_rstar = {}         # uid -> (K,) anchor similarity profile, CONSTANT
    gram_mu = None          # (1, D) frozen anchor mean, CONSTANT
    gram_missing = 0

    @torch.no_grad()
    def _build_gram_anchor():
        """Forward the FULL train pool under EMA weights, no augmentation."""
        _t0 = time.time()
        _raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict(model))
        model.eval()
        _was = model.want_trunk
        model.want_trunk = True
        # train=False: the anchor must not be an augmented view. Same studies,
        # same order, deterministic.
        # Fewer workers than tr_dl: this runs INSIDE the tr_dl loop, whose
        # persistent workers are still alive, and the job has 8 CPUs.
        _dl = DataLoader(mk(tr, False), batch_size=args.bs, shuffle=False,
                         num_workers=min(args.workers, 4), pin_memory=True)
        _uids, _H = [], []
        for _b in _dl:
            with torch.autocast("cuda", dtype=amp_dtype):
                model(_b["vol"].to(device, non_blocking=True),
                      _b["stype"].to(device), _b["mask"].to(device),
                      _b["seq"].to(device))
            _H.append(model._trunk_rep.float().cpu())
            _uids.extend(_b["study"])
        model._trunk_rep = None
        model.want_trunk = _was
        model.load_state_dict(_raw)
        model.train()
        if args.freeze_bn:
            freeze_bn_modules(model)
        _raw_H = torch.cat(_H)                              # (N, D) unnormalised
        assert len(_uids) == _raw_H.shape[0] == len(set(_uids)), \
            f"anchor pass: {len(_uids)} uids, {_raw_H.shape[0]} reps, " \
            f"{len(set(_uids))} unique - duplicate or dropped study"
        # CENTERING, and it is load-bearing. The first probe measured mean
        # |r*| = 0.9233 on UNCENTERED reps: pooled CNN features share a large
        # common component, so every pairwise cosine sat near 0.92 and the
        # penalty would have been dominated by preserving that offset - i.e.
        # exactly the coherent-global-shift over-constraint PREREG amendment 2
        # names as this form's failure mode, rather than relational structure.
        # mu* is frozen at the anchor and applied identically at train time;
        # a drifting mean would make the two sides incomparable.
        _mu = _raw_H.mean(0, keepdim=True)                  # (1, D) CONSTANT
        H = torch.nn.functional.normalize(_raw_H - _mu, dim=-1)
        _k = min(args.gram_escrow_k, H.shape[0])
        _idx = np.random.RandomState(0).choice(H.shape[0], _k, replace=False)
        _href = H[_idx].to(device)                          # (K, D)
        _rs = {u: r for u, r in zip(_uids, (H.to(device) @ _href.T).cpu())}
        _all = torch.stack(list(_rs.values()))
        print(f"[gramesc] anchor built from EMA weights over {H.shape[0]} "
              f"studies in {time.time() - _t0:.0f}s; K={_k} reference studies "
              f"(RandomState(0)); D={H.shape[1]}. CENTERED on frozen mu*: "
              f"mean |r*| {float(_all.abs().mean()):.4f} "
              f"(uncentred was 0.9233 - see the centering note in code); "
              f"r* range [{float(_all.min()):.3f}, {float(_all.max()):.3f}], "
              f"SD {float(_all.std()):.4f}", flush=True)
        return _href, _rs, _mu.to(device)

    def _gram_penalty(h, studies):
        """|| rhat_i - r*_i ||^2 / K, averaged over the micro-batch.

        Relational, not feature-pinning: h is L2-normalised first, so only the
        study's SIMILARITY PROFILE to the fixed reference set is constrained,
        never its magnitude (which the readout's LayerNorm discards anyway).
        Returns (loss or None, n_missing)."""
        _rows, _tgt, _miss = [], [], 0
        for _j, _s in enumerate(studies):
            _r = gram_rstar.get(_s)
            if _r is None:
                _miss += 1
                continue
            _rows.append(_j)
            _tgt.append(_r)
        if not _rows:
            return None, _miss
        # Same frozen mu* the anchor used - see the centering note there.
        _hn = torch.nn.functional.normalize(h[_rows].float() - gram_mu, dim=-1)
        _r = _hn @ gram_href.T                              # (n, K)
        _t = torch.stack(_tgt).to(device)
        return ((_r - _t) ** 2).sum(-1).mean() / gram_href.shape[0], _miss

    def _bagauc_gc_weight(b, y, ep, i):
        _w = lw * b["aw"].to(device) if "aw" in b else lw
        if "fw" in b:
            _w = _w * b["fw"].to(device)
        _pool_off = False
        if cotrain_sched is not None:
            _ef = ep + i / max(len(tr_dl), 1)
            if not (cotrain_sched[0] <= _ef < cotrain_sched[1]):
                _pg = torch.tensor(
                    [0.0 if s in cotrain else 1.0 for s in b["study"]],
                    device=device)
                if _pg.min() < 1.0:
                    _w = _w * _pg.unsqueeze(-1)
                    _pool_off = bool(_pg.max() < 1.0)
        if hard_neg:
            hw = torch.ones_like(y)
            for ti, gi, mult, nrm in hard_neg:
                hw[:, ti] = torch.where(
                    y[:, gi] >= 0.5, torch.full_like(y[:, gi], mult),
                    torch.ones_like(y[:, gi])) / nrm
            _w = _w * hw
        _w_full = _w * torch.ones_like(y)
        _grade_np = np.stack([
            bagauc_grade.get(s, np.full(len(FINDINGS), -99, np.int64))
            for s in b["study"]])
        _grade_t = torch.from_numpy(_grade_np).to(device)
        return _w, _w_full, _grade_t, _pool_off

    def _bagauc_gc_local_loss(b, y, out, logits, aux, _w, _w_full, _grade_t,
                              _pool_off, ep, i):
        """Every loss term EXCEPT the deferred bagauc pairwise surrogate,
        which the caller adds separately using pass B's G. Mirrors the
        non-gradcache per-microstep block verbatim."""
        _sh = None
        if args.rep_gls_sharp > 0:
            _sh = (((y - _ladder_lo).abs() < 1e-6)
                   | ((y - _ladder_hi).abs() < 1e-6)
                   ).to(y.dtype) * args.rep_gls_sharp
        # bagauc is always active on this path (caller only calls it when
        # bagauc_active) - zero the BCE weight on every cell with a known
        # grade-bag; the genuine cross-bag gradient for those cells arrives
        # entirely via the surrogate the caller adds after this returns.
        _w_bce = _w_full.clone()
        _w_bce[_grade_t != -99] = 0.0
        if "lr_interval" in b:
            loss = batch_label_bce(
                logits, y, b, weight=_w_bce, sharpen=_sh)
        else:
            loss = soft_bce(logits, y, weight=_w_bce, sharpen=_sh)
        if pairmax:
            hp = b["has_pairmax"].to(device) if "has_pairmax" in b \
                else torch.zeros(len(logits), device=device)
            if hp.sum() > 0:
                zp = torch.max(logits[:, 2], logits[:, 3])
                lp = torch.nn.functional.binary_cross_entropy_with_logits(
                    zp, b["y_pairmax"].to(device), reduction="none")
                loss = loss + args.pair_max_w * (
                    (lp * hp).sum() / hp.sum().clamp(min=1))
        if comp_pairs and args.comp_pair_w > 0:
            _cp = [(ci, lo, hi) for ci, u in enumerate(b["study"])
                   for lo, hi in comp_pairs.get(u, ())]
            if _cp:
                _bi = torch.tensor([c[0] for c in _cp], device=device)
                _lo = torch.tensor([c[1] for c in _cp], device=device)
                _hi = torch.tensor([c[2] for c in _cp], device=device)
                _z = logits.float()
                loss = loss + args.comp_pair_w * torch.nn.functional.softplus(
                    _z[_bi, _lo] - _z[_bi, _hi]).mean()
        if text_dim:
            tv = torch.from_numpy(
                np.stack([text_vec.get(u, np.zeros(text_dim, np.float32))
                          for u in b["study"]])).to(device)
            have = (tv.abs().sum(-1) > 0).float()
            cos = (out[3].float() * tv).sum(-1)
            loss = loss + args.text_w * (
                ((1 - cos) * have).sum() / have.sum().clamp(min=1))
        m = b["mask"].to(device).unsqueeze(-1)
        _wa = _w.unsqueeze(1) if _w.dim() == 2 else _w
        if "lr_interval" in b:
            aux_loss = _label_repair_aux_loss(
                aux, y, m, _wa, b, b["stype"].to(device))
        elif args.auxbag:
            aux_loss = _aux_bag_loss(aux, y, m, _wa)
        else:
            aux_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                aux, y.unsqueeze(1).expand_as(aux), reduction="none")
                * m * _wa).sum() / m.sum().clamp(min=1) / y.shape[1]
            if _vis_tab is not None:
                aux_loss = _aux_vis_loss(
                    aux, y, m, _wa, b["stype"].to(device))
        loss = loss + args.aux_w * aux_loss
        if args.rank_w > 0 and rank_bank:
            bl_ = torch.cat([b_[0] for b_ in rank_bank])
            by_ = torch.cat([b_[1] for b_ in rank_bank])
            rl, n_pair = 0.0, 0
            for j in range(y.shape[1]):
                cur_y = y[0, j]
                if float(lw[j]) == 0.0:
                    continue
                if cur_y > 0.60:
                    partner = bl_[by_[:, j] < 0.40, j]
                    if partner.numel():
                        rl = rl + float(lw[j]) * F.softplus(
                            -(logits[0, j] - partner)).mean()
                        n_pair += 1
                elif cur_y < 0.40:
                    partner = bl_[by_[:, j] > 0.60, j]
                    if partner.numel():
                        rl = rl + float(lw[j]) * F.softplus(
                            -(partner - logits[0, j])).mean()
                        n_pair += 1
            if n_pair:
                loss = loss + args.rank_w * (rl / n_pair)
        if aux_names:
            ya = b["y_aux"].to(device)
            hv = (b["has_aux"].to(device).unsqueeze(-1)
                  * torch.isfinite(ya).float())
            ra = (torch.nn.functional.binary_cross_entropy_with_logits(
                out[-1], torch.nan_to_num(ya), reduction="none") * hv)
            loss = loss + args.report_aux_w * (
                ra.sum() / hv.sum().clamp(min=1))
        if args.pair_route_reg > 0:
            at = out[2].float()
            tv2 = 0.0
            for fa, fb in ((2, 3), (4, 5)):
                tv2 = tv2 + 0.5 * (at[:, fa] - at[:, fb]).abs().sum(-1).mean()
            loss = loss + args.pair_route_reg * (tv2 / 2)
        if _gset and b["study"][0] in _gset:
            loss = loss * args.gold_train_w
        if args.rank_w > 0 and not _pool_off:
            rank_bank.append((logits.detach().float(), y.detach().float()))
            if len(rank_bank) > args.rank_bank:
                rank_bank.pop(0)
        if dann_head is not None:
            _dt = torch.tensor([dom_of.get(s, 3) for s in b["study"]],
                               device=device)
            if args.dann > 0:
                _p = (ep * len(tr_dl) + i) / max(1, args.epochs * len(tr_dl))
                _lam = args.dann * (2.0 / (1.0 + math.exp(-10.0 * _p)) - 1.0)
                _dl = torch.nn.functional.cross_entropy(
                    dann_head(_grl(model._study_vec.float(), _lam)), _dt)
                loss = loss + _dl
            else:
                _dl = torch.nn.functional.cross_entropy(
                    dann_head(model._study_vec.float()), _dt)
                loss = loss + args.site_aux * _dl
            if i % 200 == 0:
                print(f"   [site-head] CE {_dl.item():.4f}", flush=True)
        if aucm_idx:
            _zf = torch.sigmoid(logits.float()[0, aucm_idx])
            _yf = y.float()[0, aucm_idx]
            _wp = (2.0 * (_yf - 0.5)).clamp(min=0)
            _wn = (2.0 * (0.5 - _yf)).clamp(min=0)
            with torch.no_grad():
                aucm_p.mul_(0.999).add_(0.001 * _wp)
            _p = aucm_p
            _la = ((1 - _p) * _wp * (_zf - aucm_a) ** 2
                   + _p * _wn * (_zf - aucm_b) ** 2
                   + 2 * aucm_alpha * (_p * _wn * _zf - (1 - _p) * _wp * _zf))
            loss = loss + args.aucm * _la.mean()
            with torch.no_grad():
                aucm_agrad += (2 * (_p * (1 - _p) * args.aucm_margin
                                    + _p * _wn * _zf.detach()
                                    - (1 - _p) * _wp * _zf.detach())
                               - 2 * _p * (1 - _p) * aucm_alpha)
            if i % 200 == 0:
                print(f"   [aucm] loss {_la.mean().item():.4f} "
                      f"alpha {[round(v, 2) for v in aucm_alpha.tolist()]}",
                      flush=True)
        return loss

    def _bagauc_gradcache_window(batches, ep, i_last, is_first_window):
        """GradCache two-pass (three physical passes) processing of one
        --bagauc-w accumulation window. Exact by construction: the window's
        pairwise term P depends on theta only through the window's (N,12)
        logits Z, so

            dL/dtheta = sum_i d(local_i)/dtheta + sum_i (dP/dz_i)(dz_i/dtheta)

        Pass A (no_grad): forward every study, collect detached logits Z and
        the (weight, grade-bag) pair the pairwise term needs - RNG state is
        saved immediately before each study's forward so pass C can replay
        it bit-for-bit.
        Pass B: Zc = Z.clone().requires_grad_(True); run the existing
        _bagauc_window_loss on Zc (a graph of size (N,12), no backbone
        involved) and backward it PLAIN (no scaler - Zc is disconnected
        from every model parameter) to read G = Zc.grad = dP/dZ.
        Pass C: for each study, restore its saved RNG state, re-forward
        WITH grad, assert the fresh logits match pass A's (else the RNG
        replay failed and G would be evaluated at the wrong point), build
        the study's local loss exactly as the non-gradcache path would,
        add the CONSTANT surrogate bagauc_w*(logits_i . G_i) - whose
        gradient w.r.t. logits_i is exactly bagauc_w*G_i, reproducing
        (dP/dz_i)(dz_i/dtheta) via the chain rule without P itself ever
        appearing in this graph - and backward immediately so the graph
        frees before the next study. Only ONE graph is alive at a time,
        vs --accum simultaneously on the retain_graph path.
        Returns (sum of local-loss .item()s [bagauc_w*surrogate excluded,
        matching what `tot` means on the non-gradcache path], N, the (12,)
        pair-count array _bagauc_window_loss reports).
        """
        N = len(batches)
        win_start_i = i_last - N + 1
        rng_states = []
        Z_list, grade_list, w_list = [], [], []
        with torch.no_grad():
            for k, b in enumerate(batches):
                rng_states.append((torch.get_rng_state(),
                                   torch.cuda.get_rng_state()))
                y = b["y"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype):
                    out = model(b["vol"].to(device, non_blocking=True),
                               b["stype"].to(device), b["mask"].to(device),
                               b["seq"].to(device))
                Z_list.append(out[0].detach().float())
                _w, _w_full, _grade_t, _pool_off = _bagauc_gc_weight(
                    b, y, ep, win_start_i + k)
                grade_list.append(_grade_t)
                w_list.append(_w_full)
        Z = torch.cat(Z_list, dim=0)
        grade_all = torch.cat(grade_list, dim=0)
        w_all = torch.cat(w_list, dim=0)

        Zc = Z.clone().requires_grad_(True)
        bl, pair_counts = _bagauc_window_loss([(Zc, grade_all, w_all)])
        if bl is not None:
            bl.backward()
            G = Zc.grad.detach().clone()
        else:
            G = torch.zeros_like(Z)

        sum_loss = 0.0
        for k, b in enumerate(batches):
            cpu_rng, cuda_rng = rng_states[k]
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            step_i = win_start_i + k
            y = b["y"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(b["vol"].to(device, non_blocking=True),
                           b["stype"].to(device), b["mask"].to(device),
                           b["seq"].to(device))
                logits, aux = out[0], out[1]
                do_assert = is_first_window or (step_i % 200 == 0)
                if do_assert:
                    _diff = (logits.detach().float()
                            - Z[k:k + 1]).abs().max().item()
                    if _diff >= 1e-3:
                        raise SystemExit(
                            f"[bagauc-gradcache] pass-A/pass-C logit "
                            f"mismatch at ep{ep} step{step_i} study "
                            f"{b['study'][0]}: max|z_C-z_A|={_diff:.6f} >= "
                            f"1e-3 - RNG replay did not reproduce pass A; G "
                            f"would be evaluated off-policy. Aborting rather "
                            f"than corrupt a funded arm.")
                _w, _w_full, _grade_t, _pool_off = _bagauc_gc_weight(
                    b, y, ep, step_i)
                loss = _bagauc_gc_local_loss(
                    b, y, out, logits, aux, _w, _w_full, _grade_t,
                    _pool_off, ep, step_i)
                surrogate = args.bagauc_w * (logits.float()[0] * G[k]).sum()
                loss_i = loss + surrogate
            scaler.scale(loss_i / args.accum).backward()
            sum_loss += loss.item()
            if tap_probes is not None and tap_feats:
                Bv, Sv, Zv = b["vol"].shape[:3]
                _msk = b["mask"].to(device).float()
                _yv = y.float()
                _wp = _w.detach().float()
                _tl = 0.0
                for si in stage_taps:
                    _f = tap_feats[si].reshape(Bv, Sv, Zv, -1).mean(2)
                    _f = ((_f * _msk.unsqueeze(-1)).sum(1)
                          / _msk.sum(1, keepdim=True).clamp(min=1))
                    _z = tap_probes[f"s{si}"](_f)
                    _bl2 = torch.nn.functional.binary_cross_entropy_with_logits(
                        _z, _yv, reduction="none")
                    _tl = _tl + (_bl2 * _wp).mean()
                tap_feats.clear()
                tap_opt.zero_grad(set_to_none=True)
                _tl.backward()
                tap_opt.step()
        if torch.cuda.is_available():
            # Cheap (one sync) diagnostic, printed every window: the whole
            # point of GradCache is only ONE backbone graph alive at a time
            # instead of --accum, so this is the number that either
            # licenses dropping --grad-checkpoint or shows the fix still
            # needs it. torch.cuda.max_memory_allocated() is a RUNNING peak
            # since process start (never reset here), so the LAST line
            # printed in a run is the true epoch/run peak.
            print(f"   [bagauc-gradcache] ep{ep} window@{win_start_i} "
                  f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
                  f"cur_mem={torch.cuda.memory_allocated()/1e9:.2f}GB",
                  flush=True)
        return sum_loss, N, pair_counts

    # ---- grad-probe setup (scratch/taskcompromise/PREREG.md) --------------
    # Targets are FLOATS: 1.15 = epoch 1, 15% of the way through its
    # optimizer steps. An integer is fraction 0.0.
    _gp_eps = {float(x) for x in args.grad_probe_epochs.replace(",", " ").split()}
    _gp_batches, _gp_names, _gp_trunk = [], [], None
    if _gp_eps:
        if not args.grad_probe_out:
            raise SystemExit("--grad-probe-epochs needs --grad-probe-out")
        os.makedirs(args.grad_probe_out, exist_ok=True)
        # SEPARATE deterministic loader: no shuffle, no augmentation, its own
        # iterator. Nothing here advances the training sampler or the global RNG,
        # which is what makes a probed run identical to an unprobed one.
        _gp_dl = DataLoader(mk(tr, False), batch_size=args.bs, shuffle=False,
                            num_workers=0, pin_memory=False)
        for _b in _gp_dl:
            _gp_batches.append(_b)
            if len(_gp_batches) >= args.grad_probe_batches:
                break
        _gp_names = [n for n, p in model.named_parameters() if p.requires_grad]
        # TRUNK = everything outside the per-finding private parameters. The
        # private head always receives its own aligned gradient even while the
        # shared representation moves against the task, so a full-model number
        # would mask exactly the effect being tested.
        _gp_trunk = [not (n.split(".")[0] in ("head", "head_b", "aux_head")
                          or "quer" in n.lower()) for n in _gp_names]
        print(f"[gradprobe] epochs={sorted(_gp_eps)} batches={len(_gp_batches)} "
              f"trunk={sum(_gp_trunk)}/{len(_gp_names)} tensors -> "
              f"{args.grad_probe_out}", flush=True)

    # ---- --ret-anchor state (scratch/retanchor/PREREG.md) -----------------
    # Built ONCE, at the end of epoch args.ret_anchor_epoch, from the EMA
    # weights. None both before that point and in every unflagged run, so
    # the batch-loop guard `ret_anchor_model is not None` costs one Python
    # `is` and the flag-off path executes no new tensor op.
    ret_anchor_model = None

    # ---- brierfork (scratch/brierfork/PREREG.md) ---------------------------
    # Every flag is argparse.SUPPRESS; with none passed _rl_mode == "bce",
    # _fork_start == 0 and _fms_k == 0, and the loop below executes exactly the
    # original statements (one extra `ep < 0` compare per epoch).
    _rl_mode = getattr(args, "report_loss", "bce")
    _rl_scale = float(getattr(args, "report_loss_scale", 1.0))
    _fms_k = int(getattr(args, "fork_measure_scale", 0))
    _fms_acc = []
    _fork_start = 0
    if _rl_mode != "bce" or _fms_k > 0:
        if not cotrain:
            raise SystemExit("--report-loss/--fork-measure-scale need a "
                             "--cotrain-pool (OAI rows keep BCE)")
        if args.bagauc_w > 0 or args.auxbag or _vis_tab is not None:
            raise SystemExit("brierfork: report-loss path not implemented "
                             "with bagauc/auxbag/aux-visibility-mask")
        if _rl_mode == "scaled" and not hasattr(args, "report_loss_scale"):
            raise SystemExit("--report-loss scaled needs --report-loss-scale")
        if _fms_k > 0 and (not hasattr(args, "fork_resume") or _rl_mode != "bce"):
            raise SystemExit("--fork-measure-scale needs --fork-resume and "
                             "--report-loss bce/absent")
        print(f"[reportloss] mode={_rl_mode} scale={_rl_scale} on report rows "
              f"(main + aux_w={args.aux_w} aux); {len(cotrain)} co-train "
              f"studies keep BCE; measure_k={_fms_k}", flush=True)

    def _rl_rowmask(studies, ndim):
        r = torch.tensor([0.0 if s in cotrain else 1.0 for s in studies],
                         device=device)
        return r.reshape((-1,) + (1,) * (ndim - 1))

    def _rl_elem(z, yt, rows, mode, force=False):
        if _sh is not None:
            raise SystemExit("brierfork: report-loss with sharpen not implemented")
        bce = F.binary_cross_entropy_with_logits(z.float(), yt.float(),
                                                 reduction="none")
        if mode == "bce":
            rep = bce
        elif mode == "brier":
            rep = 2.0 * (torch.sigmoid(z.float()) - yt.float()) ** 2
        else:
            rep = _rl_scale * bce
        return rep if force else torch.where(rows > 0, rep, bce)

    if getattr(args, "fork_resume", ""):
        _fk = torch.load(args.fork_resume, map_location="cpu", weights_only=False)
        _fx = {"fork_save_epoch", "fork_resume", "report_loss",
               "report_loss_scale", "fork_measure_scale", "stop_epoch", "out",
               "workers", "max_steps"}
        _pa, _ma = _fk["args"], vars(args)
        _bad = sorted(k for k in (set(_pa) | set(_ma)) - _fx
                      if _pa.get(k, "<absent>") != _ma.get(k, "<absent>"))
        if _bad:
            raise SystemExit("--fork-resume args mismatch vs prefix: " + "; ".join(
                f"{k}: prefix {_pa.get(k, '<absent>')!r} vs run "
                f"{_ma.get(k, '<absent>')!r}" for k in _bad))
        model.load_state_dict(_fk["model"], strict=True)
        opt.load_state_dict(_fk["opt"])
        sched.load_state_dict(_fk["sched"])
        scaler.load_state_dict(_fk["scaler"])
        if ema is not None:
            ema.shadow = {k: v.to(device) for k, v in _fk["ema_shadow"].items()}
            ema.n = _fk["ema_n"]
            for _x, _xs in zip(ema_extra, _fk["ema_extra"]):
                _x["ema"].shadow = {k: v.to(device) for k, v in _xs["shadow"].items()}
                _x["ema"].n = _xs["n"]
                _x["best"] = -1.0
        _opt_steps = _fk["opt_steps"]
        random.setstate(_fk["rng_py"])
        np.random.set_state(_fk["rng_np"])
        torch.set_rng_state(_fk["rng_torch"])
        if torch.cuda.is_available() and _fk.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(_fk["rng_cuda"])
        _fork_start = int(_fk["epoch"]) + 1
        print(f"[fork-resume] {args.fork_resume}: state after ep{_fk['epoch']} "
              f"(opt_steps={_opt_steps}, ema_n={_fk['ema_n']}, sched.last_epoch="
              f"{sched.last_epoch}, lr={[g['lr'] for g in opt.param_groups][:2]}); "
              f"continuing at ep{_fork_start}; best-trackers reset", flush=True)
        del _fk

    for ep in range(args.epochs):
        if ep < _fork_start:
            continue
        # --pfrel spec s10: ep0-1 freeze ConvNeXt stem + stages 0-2 (train the
        # projection, queries, relation block, Conv1D stack and head on a
        # fixed proven trunk); ep2+ unfreeze stages 2-3. The remaining LLRD
        # ladder is the REFERENCE recipe's (--lr / --backbone-lr-mult /
        # --llrd), not a new optimiser. Off-arm: this block never runs.
        if _pfrel_mode != "off":
            # ep0-1: only stage 4 (timm "stages.3") trains. ep2+: stages 3-4
            # ("stages.2", "stages.3"). Stem + stages 1-2 stay frozen for the
            # whole 10 epochs - spec s10 makes the full-encoder unfreeze
            # conditional on saturation, and V0 does not test saturation.
            _open = ("stages.3.",) if ep < 2 else ("stages.2.", "stages.3.")
            _n_f = 0
            for _n, _p in model.backbone.named_parameters():
                _want = _n.startswith(_open) or _n.startswith("norm_pre")
                _p.requires_grad_(bool(_want))
                _n_f += 0 if _want else 1
            print(f"[pfrel] ep{ep}: backbone frozen tensors={_n_f} "
                  f"trainable stages={_open}", flush=True)
        if _bone_mode != "off":
            # Same ladder as pfrel: ep0-1 only stage 4 ("stages.3") trains;
            # ep2+ stages 3-4. Stem + stages 1-2 stay frozen for the 10 epochs.
            _open = ("stages.3.",) if ep < 2 else ("stages.2.", "stages.3.")
            _n_f = 0
            for _n, _p in model.backbone.named_parameters():
                _want = _n.startswith(_open) or _n.startswith("norm_pre")
                _p.requires_grad_(bool(_want))
                _n_f += 0 if _want else 1
            print(f"[bone] ep{ep}: backbone frozen tensors={_n_f} "
                  f"trainable stages={_open}", flush=True)
        _vis_acc[0] = _vis_acc[1] = 0.0
        _wit_ep = [0, 0.0, 0]     # scratch/witness: [batches, sum L, cells]
        teacher_steps = 0
        teacher_rows = 0
        teacher_cells = 0
        teacher_weight_sum = 0.0
        teacher_w_start = teacher_curriculum_weight(
            args.teacher_curriculum, float(ep), args.teacher_max_w,
            args.teacher_ramp_start, args.teacher_ramp_end)
        teacher_w_end = teacher_curriculum_weight(
            args.teacher_curriculum, float(ep + 1), args.teacher_max_w,
            args.teacher_ramp_start, args.teacher_ramp_end)
        if teacher_targets is not None:
            print(f"[teacher-sched] ep{ep}: weight "
                  f"{teacher_w_start:.6f}->{teacher_w_end:.6f} "
                  f"(live micro-step binding)", flush=True)
        # Micro-batch indices at which to snapshot this epoch. Recomputed per
        # epoch because len(tr_dl) can differ between epochs when a cotrain
        # pool is resampled.
        _sub_at = {}
        if _sub_fracs and ep in _sub_eps and ema is not None:
            for _f in _sub_fracs:
                _sub_at[int(round(_f * len(tr_dl)))] = _f
        model.train()
        if args.freeze_bn:
            _nbn = freeze_bn_modules(model)
            _bn_live = sum(1 for m in model.backbone.modules()
                           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                           and m.training)
            assert _bn_live == 0, f"--freeze-bn: {_bn_live} BN modules still training"
            print(f"[freeze-bn] ep{ep}: {_nbn} BatchNorm modules held in eval() "
                  f"(running stats frozen, affine trainable); model.training="
                  f"{model.training}", flush=True)
        if cotrain_sched is not None:
            _a0 = cotrain_sched[0] <= ep < cotrain_sched[1]
            _a1 = cotrain_sched[0] <= ep + 1 - 1e-9 < cotrain_sched[1]
            print(f"[cotrain-sched] ep{ep}: pool "
                  + ("ACTIVE" if _a0 and _a1 else
                     "OFF" if not _a0 and not _a1 else
                     "flips mid-epoch"), flush=True)
        # Per-finding loss weights for THIS epoch. Deliberately NOT renormalised.
        # soft_bce reduces with loss.mean() over a FIXED 12 columns, so dropping
        # one term does not rescale the survivors - measured, the other eleven
        # findings' gradients are bit-identical to an unmuted run. An earlier
        # version divided by lw.mean() "so muting is not a stealth learning-rate
        # cut", which had it exactly backwards: that line INTRODUCED a 12/11 =
        # 9.09% gradient scaling on every surviving finding. Harmless while the
        # whole loss is uniformly scaled and AdamW is scale-invariant, but it
        # silently upweights main-vs-auxiliary by 9% the moment --report-aux or
        # --text-w is on, and it interacts with grad-norm clipping.
        lw = torch.ones(len(FINDINGS), device=device)
        for i, (e, w) in mute.items():
            if ep >= e:
                lw[i] = w
        if mute:
            print(f"[mute] epoch {ep} weights: "
                  + " ".join(f"{f.split()[0][:4]}:{v:.2f}"
                             for f, v in zip(FINDINGS, lw.tolist())), flush=True)
        if args.mmix_p > 0:
            print(f"[regbundle] epoch {ep}: {mmix_steps} mixed steps so far",
                  flush=True)
        # ---- grad-probe: which OPTIMIZER-STEP indices fire this epoch -----
        # Targets are `epoch + fraction`, and the fraction is resolved against
        # OPTIMIZER-STEP progress, not micro-batch index, so that with grad
        # accumulation "15% through the epoch" means the first genuine update
        # at or after that fraction. An integer target is fraction 0.0, so a
        # sub-epoch measurement means exactly what an epoch-boundary one does.
        _gp_g, _gp_armed, _gp_fire = None, False, {}
        _gp_nsteps = max(len(tr_dl) // max(args.accum, 1), 1)
        for _t in _gp_eps:
            if int(_t) == ep:
                _gp_fire[int(round((_t - int(_t)) * _gp_nsteps))] = _t
        if _gp_fire:
            print(f"[gradprobe] ep{ep}: {_gp_nsteps} opt-steps, firing at "
                  f"{sorted(_gp_fire)} -> targets {sorted(_gp_fire.values())}",
                  flush=True)
        _gp_step_i = 0

        t0, tot, n = time.time(), 0.0, 0
        _clip_norms = []
        opt.zero_grad(set_to_none=True)
        bagauc_active = args.bagauc_w > 0 and ep >= args.bagauc_start_ep
        # --silent-mute-from. Resolved per MICRO-STEP (not per epoch) so a
        # fractional activation point like 0.30 lands where the sub-epoch EMA
        # snapshot does, which is what makes the within-run delta in
        # scratch/silmute/PREREG.md 4 a clean before/after on one run.
        silmute_on = bool(silmute_grade) and (
            ep + 1.0 > args.silent_mute_from)
        silmute_cell_counts = np.zeros(len(FINDINGS), dtype=np.int64)
        silmute_unresolved = 0
        silmute_studies_seen = 0
        rcw_cells_zeroed = 0
        # --distill-logit-mse execution audit, same discipline as silmute's:
        # count the cells the term ACTUALLY reached, per epoch, and abort if a
        # flagged run ever reaches zero.
        lmse_cells = 0
        lmse_steps = 0
        lmse_sum = 0.0
        # --aux-seg-w execution audit, same discipline: count the (series,
        # slice) windows the term actually reached, per epoch.
        auxseg_series = 0
        auxseg_steps = 0
        auxseg_sum = 0.0
        moaks_cells = 0
        moaks_steps = 0
        moaks_sum = 0.0
        retanchor_steps = 0
        retanchor_sum = 0.0
        bilat_apps = 0
        bilat_pen_sum = 0.0
        bilat_mass = {s: 0.0 for s in BILAT_STRATA}
        bilat_terms = {s: 0 for s in BILAT_STRATA}
        if _wd_bb_groups:
            print(f"   [wdecay] ep{ep}: {args.wd_target} groups {_wd_bb_groups} wd = "
                  f"{[opt.param_groups[gi]['weight_decay'] for gi in _wd_bb_groups]} "
                  f"| others {[g['weight_decay'] for gi,g in enumerate(opt.param_groups) if gi not in _wd_bb_groups]}",
                  flush=True)
        gramesc_on = (args.gram_escrow_w > 0 and args.gram_escrow_from >= 0)
        model.want_trunk = gramesc_on
        gram_n, gram_sum, gram_probe_rows, gram_missing = 0, 0.0, [], 0
        if bool(silmute_grade):
            print(f"   [silmute] ep{ep}: silent BCE weight x"
                  f"{args.silent_mute_w} from frac "
                  f"{max(args.silent_mute_from - ep, 0.0):.2f} "
                  f"({'ACTIVE' if silmute_on else 'off'})", flush=True)
        bagauc_buf = []
        bagauc_gc_buf = []
        bagauc_pair_counts = np.zeros(len(FINDINGS), dtype=np.int64)
        bagauc_n_windows = 0
        for i, b in enumerate(tr_dl):
          # --main-order-hash (diagnostic, PREREG_dose.md G3): reads only the
          # study-id strings of the first N main micro-batches of the first
          # epoch run; no RNG, no tensor op. Off (0) = never entered.
          if args.main_order_hash > 0 and _moh_state[0] is None:
              _moh_state[1].extend(b["study"])
              if len(_moh_state[1]) >= args.main_order_hash:
                  import hashlib
                  _moh = _moh_state[1][:args.main_order_hash]
                  _moh_state[0] = hashlib.sha1(
                      "\n".join(_moh).encode()).hexdigest()
                  print(f"[main-order] first {len(_moh)} main uids sha1 "
                        f"{_moh_state[0]} (ep{ep}; first {_moh[0]} last "
                        f"{_moh[-1]})", flush=True)
          # --max-steps: checked BEFORE the forward so that 0 does no training
          # at all and the epoch-end validation below still runs on the loaded
          # weights (the equivalence gate).
          if args.max_steps >= 0 and _opt_steps >= args.max_steps:
              _steps_done = True
              print(f"[max-steps] reached {_opt_steps} optimizer steps at "
                    f"micro-batch {i} of epoch {ep}; stopping training and "
                    f"running the epoch-end validation/save", flush=True)
              _evform_activity(f" max-steps ep{ep}")
              break
          # --wd-from: flip the backbone group's decay ONCE, at the first
          # micro-step past the activation point. Done here rather than at the
          # epoch boundary so the fractional point is honoured exactly, and it
          # is a pure optimizer-state change — no graph, no forward, so it is
          # safe anywhere in the step.
          if (_wd_bb_groups
                  and opt.param_groups[_wd_bb_groups[0]]["weight_decay"] != args.wd * args.wd_scale
                  and (ep + i / max(len(tr_dl), 1)) >= args.wd_from):
              _old = opt.param_groups[_wd_bb_groups[0]]["weight_decay"]
              for _gi in _wd_bb_groups:
                  opt.param_groups[_gi]["weight_decay"] = args.wd * args.wd_scale
              print(f"   [wdecay] ep{ep}+{i/max(len(tr_dl),1):.2f}: {args.wd_target} "
                    f"groups {_wd_bb_groups} weight_decay {_old} -> "
                    f"{args.wd * args.wd_scale}", flush=True)
          # --gram-escrow anchor, built lazily on the FIRST micro-step past
          # --gram-escrow-from. It MUST happen here, before this step's
          # forward: _build_gram_anchor swaps EMA weights in via
          # load_state_dict, which copies IN PLACE, so doing it after the
          # forward would mutate the very parameter tensors this step's
          # backward still needs and silently corrupt the gradient.
          if (args.gram_escrow_w > 0 and args.gram_escrow_from >= 0
                  and gram_href is None
                  and (ep + i / max(len(tr_dl), 1)) >= args.gram_escrow_from):
              gram_href, gram_rstar, gram_mu = _build_gram_anchor()
          # --bagauc-gradcache: defer this micro-step's forward/loss/backward
          # to the window boundary below (_bagauc_gradcache_window) instead
          # of doing it here - this whole `else` block, byte-for-byte
          # unchanged, is exactly what runs when the flag is off (bagauc off
          # entirely, or bagauc on without gradcache).
          _gc_defer = bagauc_active and args.bagauc_gradcache
          if _gc_defer:
              bagauc_gc_buf.append(b)
              loss = None
          else:
            y = b["y"].to(device, non_blocking=True)
            if _sw_tap is not None:
                _sw_tap.arm()
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(b["vol"].to(device, non_blocking=True),
                            b["stype"].to(device), b["mask"].to(device),
                            b["seq"].to(device))
                logits, aux = out[0], out[1]
                _sw_e = (_sw_tap.take(b["mask"].shape[0], b["mask"].shape[1],
                                      args.n_slices)
                         if _sw_tap is not None else None)
                # Per-cell confidence multiplies the per-finding weight. At
                # bs=1 "aw" is (1,12) and broadcasts against lw's (12,).
                _w = lw * b["aw"].to(device) if "aw" in b else lw
                # Mirrored studies carry labels that no longer describe the
                # image for the laterality-specific findings with no pair -
                # a mirrored MCL sits where the LCL belongs. Mask those rather
                # than teach them wrong. Applies to the auxiliary per-series
                # head too, below, via the same _w.
                if "fw" in b:
                    _w = _w * b["fw"].to(device)
                # Live teacher curriculum. The dataset still supplies the
                # report/OAI target in b["y"]; only this local tensor is
                # blended, at the actual micro-step, so persistent workers
                # and validation loaders remain untouched. Missing teacher
                # rows (including all gold rows) retain the report target.
                _teacher_lam = teacher_curriculum_weight(
                    args.teacher_curriculum,
                    ep + i / max(len(tr_dl), 1),
                    args.teacher_max_w, args.teacher_ramp_start,
                    args.teacher_ramp_end)
                if teacher_targets is not None and _teacher_lam > 0.0:
                    _trows = [s for s in b["study"]
                              if s in teacher_train_uids]
                    if _trows:
                        _tm_np = np.asarray(
                            [s in teacher_train_uids for s in b["study"]],
                            dtype=bool)
                        _ty_np = np.stack([
                            teacher_targets[s] if s in teacher_train_uids
                            else np.zeros(len(FINDINGS), dtype=np.float32)
                            for s in b["study"]]).astype(np.float32)
                        _tm = torch.from_numpy(_tm_np).to(device)
                        _ty = torch.from_numpy(_ty_np).to(device)
                        _ym = ((1.0 - _teacher_lam) * y
                               + _teacher_lam * _ty)
                        y = torch.where(_tm.unsqueeze(-1), _ym, y)
                        teacher_steps += 1
                        teacher_rows += int(_tm_np.sum())
                        teacher_cells += int(_tm_np.sum()) * len(FINDINGS)
                        teacher_weight_sum += (_teacher_lam *
                                               float(_tm_np.sum()))
                # B3 gate: outside [LO,HI) a pool study's weight is zeroed -
                # exactly equivalent to excluding it (LayerNorm-only stack, no
                # batch statistics; grads average over a FIXED accum divisor
                # either way) while every step/EMA/snapshot index stays
                # byte-identical to the constant arm.
                _pool_off = False
                if cotrain_sched is not None:
                    _ef = ep + i / max(len(tr_dl), 1)
                    if not (cotrain_sched[0] <= _ef < cotrain_sched[1]):
                        _pg = torch.tensor(
                            [0.0 if s in cotrain else 1.0 for s in b["study"]],
                            device=device)
                        if _pg.min() < 1.0:
                            _w = _w * _pg.unsqueeze(-1)
                            _pool_off = bool(_pg.max() < 1.0)
                if hard_neg:
                    # Concentrate a finding's gradient on the studies that
                    # supply its HARD negatives. Measured motivation, not a
                    # guess: across 80 arms the Contusion head scores the
                    # FRACTURE label better than the Fracture head does
                    # (0.8741 vs 0.8662, winning 50/80), so the Fracture column
                    # is being served by a marrow-oedema detector and its own
                    # head contributes -0.008. 20.5% of Fracture's ranking
                    # pairs put it against an oedema-positive non-fracture, and
                    # that subgroup is where the whole column is decided.
                    #
                    # Divided by a PRECOMPUTED normaliser so the column's
                    # expected total weight is unchanged and only its
                    # DISTRIBUTION across studies moves. Without that this
                    # would be "train Fracture harder", which is confounded
                    # with the mechanism - and a uniform per-finding rescale
                    # cannot change a rank metric anyway.
                    hw = torch.ones_like(y)
                    for ti, gi, mult, nrm in hard_neg:
                        hw[:, ti] = torch.where(y[:, gi] >= 0.5,
                                                torch.full_like(y[:, gi], mult),
                                                torch.ones_like(y[:, gi])) / nrm
                    _w = _w * hw
                # --silent-mute-from: scale the soft-BCE weight on EXPLICITLY
                # SILENT cells (report grade -1). Placed after the cotrain
                # gate and hard-neg so it composes multiplicatively with every
                # existing weight, and gated on the FRACTIONAL epoch so the
                # activation point coincides with the ep0.30 sub-epoch EMA
                # snapshot the within-run read anchors on. Grade -99 (a
                # probs-only row with no 'grades' key) is never touched:
                # absent is not silent. Unflagged arms never enter this block.
                if silmute_on and (
                        ep + i / max(len(tr_dl), 1)) >= args.silent_mute_from:
                    _sm_np = np.stack([
                        silmute_grade.get(s, np.full(len(FINDINGS), -99,
                                                     np.int64))
                        for s in b["study"]])
                    _sm_mask = torch.from_numpy(_sm_np == -1).to(device)
                    _w = _w * torch.where(
                        _sm_mask,
                        torch.full_like(y, args.silent_mute_w),
                        torch.ones_like(y))
                    silmute_cell_counts += (_sm_np == -1).sum(0)
                    # UNRESOLVED counter. A study id absent from the labels
                    # file yields -99 and is silently NOT muted - the same
                    # shape as the name-mismatch bug that twice dropped roster
                    # members from the notebook without any error. If this is
                    # nonzero the arm is under-dosed and the read is invalid,
                    # so it is counted rather than assumed away.
                    silmute_unresolved += int((_sm_np == -99).all(1).sum())
                    silmute_studies_seen += len(b["study"])
                # --report-cell-weight: scale named findings' cells on REPORT
                # rows only (OAI rows keep weight 1). Composes multiplicatively
                # after silmute so both can coexist; off (default) = no op.
                if rcw_col is not None:
                    _rc_np = np.array([s in rcw_report for s in b["study"]])
                    _rc_mask = torch.from_numpy(_rc_np).to(device)[:, None]
                    _w = _w * torch.where(
                        _rc_mask.expand_as(y),
                        rcw_col_t.to(y.dtype).unsqueeze(0).expand_as(y),
                        torch.ones_like(y))
                    if rcw_cells_zeroed == 0 and _rc_np.any():
                        print(f"   [rcw] first fire ep{ep} step {i}: report rows "
                              f"in batch {int(_rc_np.sum())}/{len(_rc_np)}; "
                              f"per-finding weight now "
                              + " ".join(f"{_f[:4]}:{float(v):.2f}" for _f, v in
                                         zip(FINDINGS, _w.reshape(-1, len(FINDINGS))[int(np.argmax(_rc_np))].tolist())),
                              flush=True)
                    rcw_cells_zeroed += int(_rc_np.sum()) * int((rcw_col == 0).sum())
                # Rep-GLS sharpening. The mask is derived from the TARGET, not
                # from a plumbed-through grade: the calibrated ladder's end
                # values are distinct floats from every silent prior, so an
                # exact match identifies "the report stated this categorically"
                # exactly. Note this deliberately EXCLUDES silent cells - the
                # ones where P(gold=1|silent) is 0.258 for Synovitis - which is
                # the whole point, and it is why a global sharpen would be a
                # different and worse experiment.
                _sh = None
                if args.rep_gls_sharp > 0:
                    _sh = (((y - _ladder_lo).abs() < 1e-6)
                           | ((y - _ladder_hi).abs() < 1e-6)
                           ).to(y.dtype) * args.rep_gls_sharp
                # --bagauc: REPLACE (not add to) soft-BCE per cell where a
                # known grade-bag exists, once active. A cell with no known
                # bag (probs-only row) keeps its BCE gradient - never a
                # fabricated bag. The genuine cross-bag pairwise term for the
                # zeroed cells is computed once per WINDOW below, at the
                # accumulation boundary, from live (non-detached) logits.
                # args.bagauc_w <= 0 (the default, and every pre-existing
                # arm) takes the ORIGINAL single line, byte-identical - no
                # extra tensor op, no possible behaviour change.
                # --mmix-p (scratch/regbundle/PREREG.md): manifold mixup on
                # the 12 pre-head query vectors against the previous study
                # (detached bank), with y / cell weight / sharpen mixed by
                # the SAME per-cell lam. Cells lacking a supervised target
                # in either study (cotrain-pool zeros, mirrored fw masks,
                # silent mutes) stay unmixed (lam_cell = 1). Off (default)
                # is the ORIGINAL code path, no extra tensor op.
                if args.mmix_p > 0:
                    _w_cell = _w * torch.ones_like(y)
                    # A cell mixes only if it is supervised (w > 0) AND
                    # finite in BOTH partners; the 12 primary targets carry
                    # no NaN (silent = a real 0.10 target; unsupervised
                    # cotrain-pool / mirrored cells are w = 0), but the
                    # isfinite guard makes a NaN partner cell fall back to
                    # lam = 1 (own value, own weight) rather than NaN-poison
                    # the loss. Amendment 2: mixing is written as
                    # torch.where so an unmixed cell takes its own tensor
                    # value EXACTLY (no 0 * NaN, no lam*y + (1-lam)*y
                    # rounding).
                    _ok = (_w_cell > 0) & torch.isfinite(y)
                    if (mmix_bank_w is not None
                            and mmix_bank_w.shape == _w_cell.shape):
                        _ok = _ok & (mmix_bank_w > 0) & torch.isfinite(mmix_bank_y)
                    else:
                        _ok = torch.zeros_like(_ok)
                    _y_raw, _w_raw, _sh_raw = y, _w_cell, _sh
                    _w_orig, _lg_raw = _w, logits
                    _lgm, _lam = model.mmix_logits(_ok)
                    if _lgm is not None:
                        _lam = _lam.to(y.dtype)
                        _mixed = _lam < 1
                        y = torch.where(_mixed,
                                        _lam * y + (1.0 - _lam) * mmix_bank_y, y)
                        _w = torch.where(_mixed,
                                         _lam * _w_cell + (1.0 - _lam) * mmix_bank_w,
                                         _w_cell)
                        if _sh is not None and mmix_bank_sh is not None:
                            _sh = torch.where(
                                _mixed,
                                _lam * _sh + (1.0 - _lam) * mmix_bank_sh, _sh)
                        logits = _lgm
                        mmix_steps += 1
                    mmix_bank_y = _y_raw.detach()
                    mmix_bank_w = _w_raw.detach()
                    mmix_bank_sh = (_sh_raw.detach() if _sh_raw is not None
                                    else None)
                # --dismix-p (scratch/dismix/PREREG.md): ordinal-adjacent
                # per-finding mixup on the stashed pre-head query vectors.
                # Bank is advanced with the RAW tensors every step; only the
                # primary soft-BCE below sees the mixed logits/targets and
                # the raw tensors are restored right after it. Off (default)
                # is the ORIGINAL code path, no extra tensor op.
                if dismix is not None:
                    _dm_raw = (y, _w, _sh, logits)
                    _dm_z = model._dismix_z
                    model._dismix_z = None
                    if _dm_z is None:
                        raise RuntimeError("dismix: forward did not stash z")
                    _dm_wc = _w * torch.ones_like(y)
                    _dm_ok = (_dm_wc > 0) & torch.isfinite(y)
                    (_dm_lg, _dm_y, _dm_w, _dm_sh, _dm_lam,
                     _dm_hit) = dismix.apply(model, _dm_z, y, _dm_wc, _sh,
                                             _dm_ok)
                    dismix.update(_dm_z, y, _dm_wc, _sh)
                    if _dm_hit:
                        y, _w, _sh, logits = _dm_y, _dm_w, _dm_sh, _dm_lg
                if args.bagauc_w <= 0:
                    if _rl_mode != "bce":
                        # brierfork: report rows get the arm's loss, OAI rows
                        # BCE; same weight/mean reduction as soft_bce.
                        _rl_rows = _rl_rowmask(b["study"], logits.dim())
                        loss = (_rl_elem(logits, y, _rl_rows, _rl_mode)
                                * _w).mean()
                    elif "lr_interval" in b:
                        loss = batch_label_bce(
                            logits, y, b, weight=_w, sharpen=_sh)
                    else:
                        loss = soft_bce(logits, y, weight=_w, sharpen=_sh)
                else:
                    _w_full = _w * torch.ones_like(y)
                    if bagauc_active:
                        _grade_np = np.stack([
                            bagauc_grade.get(s, np.full(len(FINDINGS), -99, np.int64))
                            for s in b["study"]])
                        _grade_t = torch.from_numpy(_grade_np).to(device)
                        _w_bce = _w_full.clone()
                        _w_bce[_grade_t != -99] = 0.0
                    else:
                        _w_bce = _w_full
                    if "lr_interval" in b:
                        loss = batch_label_bce(
                            logits, y, b, weight=_w_bce, sharpen=_sh)
                    else:
                        loss = soft_bce(logits, y, weight=_w_bce, sharpen=_sh)
                # --- evdrop masked-view loss (scratch/evdrop/PREREG.md) ----
                # The model stashed the masked-view study logits during THIS
                # forward (None on the 1-p of forwards that did not fire, and
                # always None in eval). Same targets, same per-cell weights,
                # same sharpen; the FOCAL mask zeroes the eleven-twelfths of
                # the cell grid the masking never touched, so the diffuse
                # columns and Synovitis receive no second gradient.
                if args.evdrop and model._evdrop_logits is not None:
                    loss = loss + soft_bce(model._evdrop_logits, y,
                                           weight=_w * _evdrop_cw,
                                           sharpen=_sh)
                    _evd_ct[0] += 1
                    if _evd_ct[0] in (1, 10, 100) or _evd_ct[0] % 2000 == 0:
                        print(f"[evdrop] masked-view loss added "
                              f"{_evd_ct[0]} times", flush=True)
                if args.evdrop:
                    model._evdrop_logits = None
                # --- rubricrung loss (default off) -------------------------
                if rung_rows:
                    _rz = model._rung_logits
                    _rt = np.zeros((len(b["study"]), len(rung_rows)), np.float32)
                    _rm = np.zeros_like(_rt)
                    for _i, _s in enumerate(b["study"]):
                        if _s in rung_tgt:
                            _rt[_i], _rm[_i] = rung_tgt[_s]
                    _rt = torch.from_numpy(_rt).to(_rz.device)
                    _rm = torch.from_numpy(_rm).to(_rz.device)
                    _rl = F.binary_cross_entropy_with_logits(
                        _rz.float(), _rt, reduction="none")
                    loss = loss + args.rung_w * (_rl * _rm).sum() / _rm.numel()
                # --- PAT patch-aggregate loss (scratch/shortcutarms §PAT) --
                # Stashed by THIS training forward (always None in eval).
                # Same targets / cell weights / sharpen as the primary loss;
                # the reference's equal-weight sum, times --pat-w.
                if args.pat and model._pat_logits is not None:
                    loss = loss + args.pat_w * soft_bce(
                        model._pat_logits, y, weight=_w * _pat_cw,
                        sharpen=_sh)
                    _pat_ct[0] += 1
                    if _pat_ct[0] in (1, 10, 100) or _pat_ct[0] % 2000 == 0:
                        print(f"[pat] patch loss added {_pat_ct[0]} times",
                              flush=True)
                if args.pat:
                    model._pat_logits = None
                # --- WITNESS-EV KL (scratch/witness/PREREG.md) -------------
                # `aux` IS out[1], the per-series evidence, in exactly the
                # space the teacher npz stores (osseous_factorize applied by
                # the forward to both). For each study and each of the 11
                # non-Synovitis findings: w_T = softmax over the PRESENT,
                # witness-covered series of (teacher evidence / tau), w_S the
                # same softmax of the STUDENT's evidence over the SAME series,
                # and the term is KL(w_T || w_S), entropy-gated. Alignment is
                # by SERIES UID (b["wit_ser"]), never by slot: the train
                # loader reshuffles the series every epoch.
                if witness is not None:
                    if "wit_ser" not in b:
                        raise RuntimeError(
                            "--witness-targets is on but the batch carries no "
                            "'wit_ser' key: the training loader was not built "
                            "from WitnessStudyDataset, so no series alignment "
                            "is possible")
                    # `y` here is the BLENDED target (teacher curriculum
                    # already applied above) and mmix/dismix are refused, so
                    # gate=entropy_conf reads exactly the cell target the
                    # primary soft-BCE just used.
                    _wl, _wk = witness_loss(
                        aux, b["mask"], b["study"], b["wit_ser"], witness,
                        args.witness_tau, args.witness_gate, _wit_cols,
                        _wit_ncol, target=y)
                    if _wk > 0:
                        loss = loss + args.witness_beta * _wl.to(loss.dtype)
                        _wit_ct[0] += 1
                        _wit_ct[1] += float(_wl.detach())
                        _wit_ct[2] += _wk
                        _wit_ep[0] += 1
                        _wit_ep[1] += float(_wl.detach())
                        _wit_ep[2] += _wk
                        if _wit_ct[0] in (1, 10, 100) \
                                or _wit_ct[0] % 2000 == 0:
                            print(f"[witness] L_witness running mean "
                                  f"{_wit_ct[1] / _wit_ct[0]:.5f} over "
                                  f"{_wit_ct[0]} batches "
                                  f"({_wit_ct[2]} gated cells)", flush=True)
                # --- SLICE-WITNESS KL (scratch/slicewit/PREREG.md) ----------
                if slicewit is not None:
                    if "wit_sli" not in b or _sw_e is None:
                        raise RuntimeError("--slicewit-targets is on but the "
                                           "batch has no wit_sli or the ev_head "
                                           "tap captured nothing")
                    _sl, _sk = slicewit_loss(
                        _sw_e, b["mask"], b["study"], b["wit_ser"],
                        b["wit_sli"], slicewit, args.slicewit_tau, _sw_cols,
                        y)
                    if _sk > 0:
                        loss = loss + args.slicewit_beta * _sl.to(loss.dtype)
                        _sw_ct[0] += 1
                        _sw_ct[1] += float(_sl.detach())
                        _sw_ct[2] += _sk
                        if _sw_ct[0] in (1, 10, 100) or _sw_ct[0] % 2000 == 0:
                            print(f"[slicewit] L_slicewit running mean "
                                  f"{_sw_ct[1] / _sw_ct[0]:.5f} over "
                                  f"{_sw_ct[0]} batches ({_sw_ct[2]} gated "
                                  f"cells)", flush=True)
                if args.mmix_p > 0:
                    # Mixing is confined to the primary soft-BCE above:
                    # restore the UNMIXED tensors for every later consumer
                    # (aux per-series loss, text, report-aux, ...).
                    y, _w, _sh, logits = _y_raw, _w_orig, _sh_raw, _lg_raw
                if dismix is not None:
                    y, _w, _sh, logits = _dm_raw
                # --pcol-w (scratch/recipe2/PREREG.md): auxiliary prototype
                # contrastive ordinal loss on the UNMIXED 12 pre-head query
                # embeddings stashed by the model this forward. Graded,
                # non-silent, weight>0 cells only. Off (default) = no op.
                if pcol_mod is not None:
                    _pz = model._pcol_z
                    model._pcol_z = None
                    _pl, _pn = pcol_mod(_pz, y, _w * torch.ones_like(y))
                    loss = loss + args.pcol_w * _pl
                    pcol_steps += 1
                    pcol_cells += _pn
                    pcol_loss_sum += float(_pl.detach())
                    if pcol_steps == 200 and pcol_cells == 0:
                        raise SystemExit("[pcol] 0 graded cells in the first "
                                         "200 studies - the arm would be inert")
                    if pcol_steps % 500 == 0:
                        print(f"[pcol] step {pcol_steps} mean cells/study "
                              f"{pcol_cells / pcol_steps:.2f} mean loss "
                              f"{pcol_loss_sum / pcol_steps:.4f}", flush=True)
                # --distill-logit-mse: ADD W*MSE(z, logit(p_teacher)) on the
                # rows whose target IS a teacher probability. `y` already holds
                # the teacher's value for those rows (the blend at
                # `[soft] ... blended N/M TRAIN targets` overwrote labels[u]),
                # so no second copy of the target file can drift out of sync
                # with what BCE is fitting. Rows outside `distill_uids` (val
                # never reaches here; gold rows carrying restored expert
                # labels; any cotrain-pool row) contribute exactly zero.
                # Guarded on the flag so an unflagged arm executes no extra
                # tensor op at all.
                if args.distill_logit_mse > 0:
                    _lm_rows = [_ri for _ri, _s in enumerate(b["study"])
                                if _s in distill_uids]
                    if _lm_rows:
                        _lm_idx = torch.tensor(_lm_rows, device=device)
                        _pt = y[_lm_idx].float().clamp(1e-4, 1.0 - 1e-4)
                        _zt = torch.log(_pt) - torch.log1p(-_pt)
                        _zs = logits[_lm_idx].float()
                        _lm = ((_zs - _zt) ** 2).mean()
                        loss = loss + args.distill_logit_mse * _lm.to(loss.dtype)
                        lmse_cells += len(_lm_rows) * len(FINDINGS)
                        lmse_steps += 1
                        lmse_sum += float(_lm.detach())
                # --condres-w (scratch/decouple_20261009/PREREG.md): learned
                # conditional-independence penalty. Guarded on the flag (an
                # argparse.SUPPRESS default), so an unflagged arm executes
                # nothing and its [args]/ck["args"] are unchanged.
                if getattr(args, "condres_w", 0.0) > 0:
                    loss = _condres_step(args, logits, y, b["study"], loss)
                # --aux-seg-w: ADD w*(CE+soft-Dice) from the decoder's
                # prediction on every (series, Z-slice) window whose series
                # carries a mask (b["seg_valid"], set by the dataset only
                # when --aux-seg-root is passed). Guarded on the flag AND on
                # "seg" being in the batch dict (only present when the
                # dataset was built with aux_seg_root truthy - see `mk` in
                # main()), so an unflagged arm executes no extra tensor op.
                if args.aux_seg_w > 0 and "seg_valid" in b:
                    _sv = b["seg_valid"].to(device, non_blocking=True)  # (B,S)
                    if _sv.sum() > 0:
                        _sy = b["seg"].to(device, non_blocking=True).long()
                        _Bv, _Sv, _Zv, _Hs, _Ws = _sy.shape
                        _sl = model._seg_logits.float()
                        _C = _sl.shape[1]
                        # model._seg_logits is (B*S*Z, C, Hs, Ws) in the SAME
                        # (B, S, Z) C-order flatten as b["vol"] / b["seg"]
                        # (KneeNet._stack_ctx: reshape(B*Z,...) from a
                        # (B*S,Z,...) tensor with S folded into the local B).
                        _sl = _sl.reshape(_Bv, _Sv, _Zv, _C, _Hs, _Ws)
                        _rows = _sv.reshape(-1) > 0
                        _sl2 = _sl.reshape(_Bv * _Sv, _Zv, _C, _Hs, _Ws)[_rows]
                        _sy2 = _sy.reshape(_Bv * _Sv, _Zv, _Hs, _Ws)[_rows]
                        _sl2 = _sl2.reshape(-1, _C, _Hs, _Ws)
                        _sy2 = _sy2.reshape(-1, _Hs, _Ws)
                        _segl = aux_seg_ce_dice(_sl2, _sy2, n_classes=_C)
                        loss = loss + args.aux_seg_w * _segl.to(loss.dtype)
                        auxseg_series += int(_rows.sum())
                        auxseg_steps += 1
                        auxseg_sum += float(_segl.detach())
                if bagauc_active:
                    # Buffer the LIVE (graph-attached) logits, the original
                    # per-cell weight (pre-zeroing, so mute/hard-neg/pool-gate/
                    # flip-mask still exclude a cell from pairing) and the
                    # grade-bag ids. NOT detached - this is exactly what makes
                    # the loss able to reorder two studies, unlike --rank-w's
                    # bank (killed arm: fixed detached partners make the
                    # update a monotone map of the study's own logit alone).
                    bagauc_buf.append((logits, _grade_t, _w_full))
                if pairmax:
                    # Unsplit "a meniscus is torn" label (MRNet) supervises the
                    # MAX of the two compartment logits - MIL over the pair, so
                    # gradient reaches only the compartment the model already
                    # rates higher, and the target is symmetric under the
                    # medial/lateral swap (flip_swap needs no special case).
                    hp = b["has_pairmax"].to(device) if "has_pairmax" in b \
                        else torch.zeros(len(logits), device=device)
                    if hp.sum() > 0:
                        zp = torch.max(logits[:, 2], logits[:, 3])
                        lp = torch.nn.functional.binary_cross_entropy_with_logits(
                            zp, b["y_pairmax"].to(device), reduction="none")
                        loss = loss + args.pair_max_w * (
                            (lp * hp).sum() / hp.sum().clamp(min=1))
                if comp_pairs and args.comp_pair_w > 0:
                    # softplus(z_lo - z_hi) inside ONE study. d/dz_lo depends on
                    # z_hi, which varies across studies, so two studies with the
                    # same z_lo get different updates and the column CAN
                    # reorder. Scanner, vendor, age and general degeneration are
                    # held constant by construction - the pair is one knee.
                    _cp = [(i, lo, hi) for i, u in enumerate(b["study"])
                           for lo, hi in comp_pairs.get(u, ())]
                    if _cp:
                        _bi = torch.tensor([c[0] for c in _cp], device=device)
                        _lo = torch.tensor([c[1] for c in _cp], device=device)
                        _hi = torch.tensor([c[2] for c in _cp], device=device)
                        _z = logits.float()
                        loss = loss + args.comp_pair_w * torch.nn.functional.softplus(
                            _z[_bi, _lo] - _z[_bi, _hi]).mean()
                if text_dim:
                    tv = torch.from_numpy(
                        np.stack([text_vec.get(u, np.zeros(text_dim, np.float32))
                                  for u in b["study"]])).to(device)
                    have = (tv.abs().sum(-1) > 0).float()
                    # 1 - cosine on unit vectors, averaged over studies that
                    # actually have an embedding.
                    cos = (out[3].float() * tv).sum(-1)
                    loss = loss + args.text_w * (
                        ((1 - cos) * have).sum() / have.sum().clamp(min=1))
                m = b["mask"].to(device).unsqueeze(-1)
                # _w is (12,) or (B,12); against the (B,S,12) aux loss the
                # (B,12) form broadcasts by PREPENDING, mapping the batch axis
                # onto the series axis - silently wrong whenever B==S, a crash
                # otherwise. Every shipped run used bs=1 (correct by luck).
                # Insert the series axis explicitly; bit-identical at bs=1.
                _wa = _w.unsqueeze(1) if _w.dim() == 2 else _w
                # The auxiliary per-series head predicts the same twelve labels,
                # so a finding muted in the main loss must be muted here too -
                # otherwise the aux path keeps teaching exactly the targets the
                # mute exists to stop teaching.
                if "lr_interval" in b:
                    aux_loss = _label_repair_aux_loss(
                        aux, y, m, _wa, b, b["stype"].to(device))
                elif args.auxbag:
                    aux_loss = _aux_bag_loss(aux, y, m, _wa)
                elif _rl_mode != "bce":
                    _rl_rows3 = _rl_rowmask(b["study"], aux.dim())
                    aux_loss = (_rl_elem(aux, y.unsqueeze(1).expand_as(aux),
                                         _rl_rows3, _rl_mode)
                                * m * _wa).sum() / m.sum().clamp(min=1) / y.shape[1]
                else:
                    aux_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                        aux, y.unsqueeze(1).expand_as(aux), reduction="none")
                        * m * _wa).sum() / m.sum().clamp(min=1) / y.shape[1]
                    if _vis_tab is not None:
                        aux_loss = _aux_vis_loss(
                            aux, y, m, _wa, b["stype"].to(device))
                if _fms_k > 0:
                    # brierfork --fork-measure-scale: param-grad norms of the
                    # REPORT-row loss (main + aux_w*aux) under BCE vs Brier at
                    # the fork weights; no optimizer step is ever taken.
                    _rr = _rl_rowmask(b["study"], logits.dim())
                    _rr3 = _rl_rowmask(b["study"], aux.dim())
                    _ya = y.unsqueeze(1).expand_as(aux)
                    _norms = {}
                    for _md in ("bce", "brier"):
                        _lm = (_rl_elem(logits, y, _rr, _md, force=True)
                               * _rr * _w).mean()
                        _la = (_rl_elem(aux, _ya, _rr3, _md, force=True)
                               * _rr3 * m * _wa).sum() / m.sum().clamp(min=1) / y.shape[1]
                        opt.zero_grad(set_to_none=True)
                        (_lm + args.aux_w * _la).backward(retain_graph=True)
                        _norms[_md] = float(torch.sqrt(sum(
                            (p.grad.float() ** 2).sum() for p in model.parameters()
                            if p.grad is not None)))
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        _p = torch.sigmoid(logits.float())
                        _gb = ((_p - y.float()).abs() * _rr * _w).sum()
                        _gr = (4 * _p * (1 - _p) * (_p - y.float()).abs()
                               * _rr * _w).sum()
                    _fms_acc.append((_norms["bce"], _norms["brier"],
                                     float(_gb), float(_gr), float(_rr.sum())))
                    if len(_fms_acc) >= _fms_k:
                        _a = np.array(_fms_acc)
                        _keep = _a[:, 4] > 0
                        _c = float(_a[_keep, 1].sum() / max(_a[_keep, 0].sum(), 1e-12))
                        _cz = float(_a[_keep, 3].sum() / max(_a[_keep, 2].sum(), 1e-12))
                        _res = {"scale_param_grad": _c, "scale_logit_grad": _cz,
                                "n_micro": int(len(_a)), "n_report_micro": int(_keep.sum()),
                                "fork_resume": args.fork_resume, "epoch": ep}
                        with open(os.path.join(args.out, "fork_scale.json"), "w") as _fh:
                            json.dump(_res, _fh, indent=1)
                        print(f"[forkscale] {json.dumps(_res)}", flush=True)
                        return
                    continue
                loss = loss + args.aux_w * aux_loss
                # --pfrel spec s9: centroid smoothness + anti-collapse
                # hinge, TOTAL weight --pfrel-aux split evenly. Only the "rel"
                # arm has centroids, so the generic control adds nothing here
                # (its _aux is an empty dict) - the term cannot flatter the
                # arm it is meant to regularise into the control's loss.
                if _pfrel_mode != "off" and args.pfrel_aux > 0:
                    _pa = getattr(model, "pfrel_net", None)
                    _pa = (_pa._aux if _pa is not None else None) or {}
                    if _pa:
                        loss = loss + (args.pfrel_aux / 2.0) * (
                            _pa["smooth"].to(loss.dtype)
                            + _pa["collapse"].to(loss.dtype))
                # --bonewsrpn: 0.2 * L_bone on the SHARED osseous axis z_E
                # (target = max(y_Cont, y_Frac), the rubric's "impact marrow
                # oedema present") + the proposal-diversity auxiliary. L_bone
                # is IDENTICAL in both arms; only the diversity term is
                # ROI-only (the dense control has no objects), exactly as in
                # pfrel. Weighted by the run's own per-cell weights for the
                # two columns, so a muted/absent cell teaches nothing.
                if _bone_mode != "off":
                    _bn = getattr(model, "bone_net", None)
                    if _bn is not None and _bn._zE is not None:
                        _wb = _w[..., 10:12] if _w.dim() == 2 else _w[10:12]
                        _wb = _wb.mean(-1).reshape(-1).float()
                        if _wb.numel() == 1:
                            _wb = _wb.expand(y.shape[0])
                        _yb = torch.maximum(y[:, 10], y[:, 11]).float()
                        _lb = torch.nn.functional.binary_cross_entropy_with_logits(
                            _bn._zE.float(), _yb, reduction="none")
                        loss = loss + 0.2 * (
                            (_lb * _wb).sum() / _wb.sum().clamp(min=1e-6)
                        ).to(loss.dtype)
                        if args.bone_aux > 0 and _bn._aux:
                            loss = loss + args.bone_aux * \
                                _bn._aux["div"].to(loss.dtype)
                if args.rank_w > 0 and rank_bank:
                    # Confidently-graded pairs only. Targets near 0.5 carry no
                    # order information - ranking them teaches noise - so the
                    # 0.60/0.40 gates keep the ambiguous middle out.
                    bl = torch.cat([b_[0] for b_ in rank_bank])   # (N,12) detached
                    by = torch.cat([b_[1] for b_ in rank_bank])
                    rl, n_pair = 0.0, 0
                    for j in range(y.shape[1]):
                        cur_y = y[0, j]
                        if float(lw[j]) == 0.0:
                            continue          # respect --mute-after
                        if cur_y > 0.60:
                            partner = bl[by[:, j] < 0.40, j]
                            if partner.numel():
                                rl = rl + float(lw[j]) * F.softplus(
                                    -(logits[0, j] - partner)).mean()
                                n_pair += 1
                        elif cur_y < 0.40:
                            partner = bl[by[:, j] > 0.60, j]
                            if partner.numel():
                                rl = rl + float(lw[j]) * F.softplus(
                                    -(partner - logits[0, j])).mean()
                                n_pair += 1
                    if n_pair:
                        # lw must apply here too. Without it a muted finding
                        # still receives ranking gradient - measured, with
                        # Synovitis muted 100% of its residual gradient came
                        # from this term, defeating --mute-after entirely.
                        loss = loss + args.rank_w * (rl / n_pair)
                if aux_names:
                    # report_aux logits are appended LAST by KneeNet.forward,
                    # after the optional text vector, so out[-1] is stable
                    # whether or not the text head exists.
                    ya = b["y_aux"].to(device)
                    # Per-CELL mask: has_aux gates the study, isfinite gates
                    # fields the label source never scores (NaN from a null
                    # grade). nan_to_num BEFORE the BCE - NaN*0 is NaN, so
                    # masking after the fact would poison the whole loss.
                    hv = (b["has_aux"].to(device).unsqueeze(-1)
                          * torch.isfinite(ya).float())
                    ra = (torch.nn.functional.binary_cross_entropy_with_logits(
                        out[-1], torch.nan_to_num(ya), reduction="none") * hv)
                    # Normalise by scored CELLS, so coverage gaps scale the
                    # loss rather than silently shrinking it. At bs=1 with an
                    # all-finite file this is bit-identical to the previous
                    # per-study normalisation (sum/1/8 == sum/8).
                    loss = loss + args.report_aux_w * (
                        ra.sum() / hv.sum().clamp(min=1))
                if moaks_names:
                    # MOAKS subregional aux loss (--moaks-aux): the same
                    # masked soft-BCE form as report-aux above - has_aux gates
                    # the study (competition rows carry no MOAKS vector),
                    # isfinite gates unscored cells (osteophytes on ~77% of
                    # knees; masked, never imputed), nan_to_num BEFORE the BCE
                    # for the same NaN*0 poisoning reason. moaks logits are
                    # appended LAST by KneeNet.forward and --report-aux is
                    # excluded up front, so out[-1] is unambiguous. Guarded on
                    # the study actually carrying targets so competition rows
                    # add no tensor op.
                    ya_m = b["y_aux"].to(device)
                    hv_m = (b["has_aux"].to(device).unsqueeze(-1)
                            * torch.isfinite(ya_m).float())
                    _nm = hv_m.sum()
                    if float(_nm) > 0:
                        ml = (torch.nn.functional
                              .binary_cross_entropy_with_logits(
                                  out[-1], torch.nan_to_num(ya_m),
                                  reduction="none") * hv_m)
                        _ml = ml.sum() / _nm.clamp(min=1)
                        loss = loss + args.moaks_aux_w * _ml
                        moaks_cells += int(_nm.item())
                        moaks_steps += 1
                        moaks_sum += float(_ml.detach())
                if ret_anchor_model is not None:
                    # --ret-anchor (scratch/retanchor/PREREG.md): label-free
                    # function preservation against the run's OWN
                    # end-of-anchor-epoch EMA state. No-grad anchor forward
                    # on the SAME (augmented) batch tensors, inside the same
                    # autocast region as the live forward; soft-BCE of the
                    # live 12 challenge-finding logits against the anchor's
                    # probabilities, on EVERY row (challenge and cotrain
                    # alike - the term never sees a label). The label losses
                    # above are untouched; aux/series/report heads are
                    # excluded (out[0]/anchor out[0] only). The anchor is
                    # eval-mode (dropout off -> deterministic target, no RNG
                    # consumed) and requires_grad=False. The second
                    # .to(device) repeats the H2D copy - accepted so the
                    # incumbent forward call above stays byte-identical.
                    # ret_anchor_model is None until the end of epoch
                    # args.ret_anchor_epoch, so the term can NEVER fire at
                    # or before the anchor epoch, and never in an unflagged
                    # run.
                    with torch.no_grad():
                        _ra_out = ret_anchor_model(
                            b["vol"].to(device, non_blocking=True),
                            b["stype"].to(device), b["mask"].to(device),
                            b["seq"].to(device))
                        _ra_tgt = torch.sigmoid(_ra_out[0].float())
                    _ra = (torch.nn.functional
                           .binary_cross_entropy_with_logits(
                               logits.float(), _ra_tgt, reduction="mean"))
                    loss = loss + args.ret_anchor_w * _ra
                    retanchor_steps += 1
                    retanchor_sum += float(_ra.detach())
                if args.pair_route_reg > 0:
                    # out[2] is the routing attention (B, n_out, S), heads
                    # averaged; padded series already carry exactly 0 weight,
                    # so they cancel inside the difference. float() because a
                    # half-precision sum over S loses the small differences
                    # this term exists to shrink.
                    at = out[2].float()
                    tv = 0.0
                    # fa/fb, NOT i/j: `i` is the enumerate variable of the batch
                    # loop, and shadowing it silently disabled the optimizer
                    # step (`(i+1) % accum`) for an entire run.
                    for fa, fb in ((2, 3), (4, 5)):   # Med/Lat Meniscus, Med/Lat OA
                        tv = tv + 0.5 * (at[:, fa] - at[:, fb]).abs().sum(-1).mean()
                    loss = loss + args.pair_route_reg * (tv / 2)
            if _gset and b["study"][0] in _gset:
                # Upweight the expert-labelled rows. At bs=1 this is a scalar on
                # the whole study's loss, applied AFTER every term so the aux and
                # ranking losses scale with it too.
                loss = loss * args.gold_train_w
            if args.rank_w > 0 and not _pool_off:
                # A gated-off pool study must not enter the ranking bank either:
                # rank partners are chosen on y alone, so an ungated append
                # would leak pool gradients around the B3 gate via other
                # studies' ranking terms.
                rank_bank.append((logits.detach().float(), y.detach().float()))
                if len(rank_bank) > args.rank_bank:
                    rank_bank.pop(0)
            if dann_head is not None:
                # After gold_train_w on purpose: the site head's strength must
                # not depend on whether this study is a gold row.
                _dt = torch.tensor([dom_of.get(s, 3) for s in b["study"]],
                                   device=device)
                if args.dann > 0:
                    # Adversarial: Ganin ramp keeps the reversal weak while
                    # features are still forming.
                    _p = (ep * len(tr_dl) + i) / max(1, args.epochs * len(tr_dl))
                    _lam = args.dann * (2.0 / (1.0 + math.exp(-10.0 * _p)) - 1.0)
                    _dl = torch.nn.functional.cross_entropy(
                        dann_head(_grl(model._study_vec.float(), _lam)), _dt)
                    loss = loss + _dl
                else:
                    # Plain supervised nuisance prediction: gradient flows
                    # FORWARD into the trunk, asking it to carry site identity
                    # as a stabilising factor rather than to erase it.
                    _dl = torch.nn.functional.cross_entropy(
                        dann_head(model._study_vec.float()), _dt)
                    loss = loss + args.site_aux * _dl
                if i % 200 == 0:
                    print(f"   [site-head] CE {_dl.item():.4f}", flush=True)
            if aucm_idx:
                # Soft masses, never a gate: a y of 0.7 is 0.4 of a positive
                # and 0 of a negative. Scores are sigmoid probs to match the
                # LibAUC formulation's [0,1] range and margin=1 default.
                _zf = torch.sigmoid(logits.float()[0, aucm_idx])
                _yf = y.float()[0, aucm_idx]
                _wp = (2.0 * (_yf - 0.5)).clamp(min=0)
                _wn = (2.0 * (0.5 - _yf)).clamp(min=0)
                with torch.no_grad():
                    aucm_p.mul_(0.999).add_(0.001 * _wp)
                _p = aucm_p
                # margin and -alpha^2 terms are constant w.r.t. model/a/b and
                # enter only alpha's manual ascent gradient below.
                _la = ((1 - _p) * _wp * (_zf - aucm_a) ** 2
                       + _p * _wn * (_zf - aucm_b) ** 2
                       + 2 * aucm_alpha * (_p * _wn * _zf - (1 - _p) * _wp * _zf))
                loss = loss + args.aucm * _la.mean()
                with torch.no_grad():
                    aucm_agrad += (2 * (_p * (1 - _p) * args.aucm_margin
                                        + _p * _wn * _zf.detach()
                                        - (1 - _p) * _wp * _zf.detach())
                                   - 2 * _p * (1 - _p) * aucm_alpha)
                if i % 200 == 0:
                    print(f"   [aucm] loss {_la.mean().item():.4f} "
                          f"alpha {[round(v, 2) for v in aucm_alpha.tolist()]}",
                          flush=True)
            # --gram-escrow. Added to `loss` here, at the very end of loss
            # assembly, so it composes with every optional term above and
            # rides the SAME backward - no second graph, no gradcache. The
            # anchor is built lazily at the first micro-step past
            # --gram-escrow-from (scratch/gramesc/PREREG.md amendment 2).
            if gramesc_on and (ep + i / max(len(tr_dl), 1)) >= args.gram_escrow_from:
                _gl, _gm = _gram_penalty(model._trunk_rep, b["study"])
                gram_missing += _gm
                if _gl is not None:
                    gram_n += 1
                    gram_sum += float(_gl.detach())
                    if args.gram_escrow_probe and \
                            len(gram_probe_rows) < args.gram_escrow_probe:
                        # lambda RULE instrument: the two gradient norms at the
                        # anchor, on the SAME micro-step, so their ratio is not
                        # contaminated by step-to-step variation.
                        _tp = [p for p in model.backbone.parameters()
                               if p.requires_grad]
                        _nrm = lambda gs: float(torch.sqrt(sum(
                            (g.float() ** 2).sum()
                            for g in gs if g is not None)))
                        _gt = _nrm(torch.autograd.grad(
                            _gl, _tp, retain_graph=True, allow_unused=True))
                        _st = _nrm(torch.autograd.grad(
                            loss, _tp, retain_graph=True, allow_unused=True))
                        gram_probe_rows.append((_st, _gt))
                        print(f"   [gramesc-probe] window {len(gram_probe_rows)}"
                              f": ||grad L_standard|| {_st:.6g}"
                              f"  ||grad L_gram|| {_gt:.6g}"
                              f"  ratio {(_st / _gt) if _gt > 0 else float('nan'):.6g}",
                              flush=True)
                        if len(gram_probe_rows) >= args.gram_escrow_probe:
                            _rat = sorted((a / b) for a, b in gram_probe_rows
                                          if b > 0)
                            _med = _rat[len(_rat) // 2]
                            print(f"[gramesc-probe] MEDIAN ratio over "
                                  f"{len(_rat)} windows = {_med:.6g}\n"
                                  f"[gramesc-probe] PRE-REGISTERED RULE "
                                  f"lambda := 0.1 * ratio = {0.1 * _med:.6g}\n"
                                  f"[gramesc-probe] write that value into "
                                  f"scratch/gramesc/PREREG.md BEFORE the "
                                  f"funded launch, then use it verbatim.",
                                  flush=True)
                            raise SystemExit(0)
                    loss = loss + args.gram_escrow_w * _gl
            if sam_state is None:
                # retain_graph: when bagauc is active this micro-step's
                # `logits` must stay live for the deferred cross-window
                # pairwise backward at the accumulation boundary below.
                scaler.scale(loss / args.accum).backward(
                    retain_graph=bagauc_active)
                # --cf-consist (scratch/cfconsist/PREREG.md): counterfactual
                # half-frame mask forward; penalise the target-logit change.
                # Run AFTER the main backward with its OWN backward so the two
                # activation graphs are never live together (both-branch form
                # OOMed a 44 GiB GPU in the smoke). The original-branch logit
                # is therefore a detached target (consistency-regularisation
                # form). SAM / bagauc / gradcache are refused at arg time, so
                # this is the only backward path. Off => never runs.
                if cf_pairs and (args.cf_prob >= 1.0
                                 or random.random() < args.cf_prob):
                    _cv = b["vol"].to(device, non_blocking=True)
                    _cst = b["stype"].to(device)
                    _cmk = b["mask"].to(device)
                    _csq = b["seq"].to(device)
                    _ch = _cv.shape[-1] // 2
                    # (B,S) series selector: present Coronal/Axial (stype 2-5)
                    _csel = ((_cst >= 2) & (_cmk > 0)).to(_cv.dtype)
                    _keep = (1.0 - _csel).view(
                        *_csel.shape, *([1] * (_cv.dim() - 2)))
                    _cf_l = 0.0
                    for _cj, _chalf in cf_pairs:
                        _v2 = _cv.clone()
                        _sl = (slice(0, _ch) if _chalf == "medial"
                               else slice(_ch, None))
                        _v2[..., _sl] = _v2[..., _sl] * _keep
                        with torch.autocast("cuda", dtype=amp_dtype):
                            _o2 = model(_v2, _cst, _cmk, _csq)
                        _z0 = logits[:, _cj].detach().float()
                        _z1 = _o2[0][:, _cj].float()
                        _d = _z0 - _z1        # > 0: masking LOWERED the target
                        _pen = (torch.relu(_d) ** 2 if args.cf_mode == "drop"
                                else _d ** 2)
                        _cf_l = _cf_l + _pen.mean()
                        cf_stat["dz"] += float(_d.detach().mean())
                        cf_stat["pos"] += int((_d.detach() > 0).sum())
                    cf_stat["n"] += 1
                    cf_stat["sum"] += float(_cf_l.detach())
                    if cf_stat["n"] % 400 == 1 or _CF_VERBOSE:
                        _nn = cf_stat["n"]
                        print(f"[cfconsist] ep{ep} it{i} n={_nn} mean_term "
                              f"{cf_stat['sum'] / _nn:.5f} mean_dz(orig-mask) "
                              f"{cf_stat['dz'] / (_nn * len(cf_pairs)):+.4f} "
                              f"frac_drop {cf_stat['pos'] / (_nn * len(cf_pairs)):.3f}",
                              flush=True)
                    scaler.scale(args.cf_consist * _cf_l / args.accum).backward()
            else:
                # SAM, m=1: THIS micro-batch's gradient alone defines the
                # ascent (kept clean by zeroing .grad between micro-batches);
                # the descent gradient at the perturbed point accumulates in
                # sam_state["acc"] and is restored into .grad at the accum
                # boundary, so unscale/clip/step/EMA below run unmodified.
                # The AMP scale factor cancels inside g/||g||, so scaled
                # grads are used directly for the perturbation; it is
                # constant within an accum window (update() only fires at
                # the boundary), so the accumulated buffers are consistent.
                scaler.scale(loss).backward()
                with torch.no_grad():
                    _gs = [p.grad for p in sam_params if p.grad is not None]
                    _nrm = torch.norm(torch.stack([g.norm() for g in _gs]))
                    _sc = args.sam_rho / (_nrm + 1e-12)
                    for p in sam_params:
                        if p.grad is not None:
                            _e = p.grad * _sc
                            p.add_(_e)
                            sam_state["e"][p] = _e
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype):
                    out2 = model(b["vol"].to(device, non_blocking=True),
                                 b["stype"].to(device), b["mask"].to(device),
                                 b["seq"].to(device))
                    logits2, aux2 = out2[0], out2[1]
                    if "lr_interval" in b:
                        loss2 = batch_label_bce(
                            logits2, y, b, weight=_w, sharpen=_sh)
                    else:
                        loss2 = soft_bce(logits2, y, weight=_w, sharpen=_sh)
                    if "lr_interval" in b:
                        aux_loss2 = _label_repair_aux_loss(
                            aux2, y, m, _wa, b, b["stype"].to(device))
                    elif args.auxbag:
                        aux_loss2 = _aux_bag_loss(aux2, y, m, _wa)
                    else:
                        aux_loss2 = (
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                aux2, y.unsqueeze(1).expand_as(aux2),
                                reduction="none")
                            * m * _wa).sum() / m.sum().clamp(min=1) / y.shape[1]
                        if _vis_tab is not None:
                            aux_loss2 = _aux_vis_loss(
                                aux2, y, m, _wa, b["stype"].to(device))
                    loss2 = loss2 + args.aux_w * aux_loss2
                if _gset and b["study"][0] in _gset:
                    loss2 = loss2 * args.gold_train_w
                scaler.scale(loss2 / args.accum).backward()
                with torch.no_grad():
                    for p in sam_params:
                        _e = sam_state["e"].pop(p, None)
                        if _e is not None:
                            p.sub_(_e)
                        if p.grad is not None:
                            if p in sam_state["acc"]:
                                sam_state["acc"][p] += p.grad
                            else:
                                sam_state["acc"][p] = p.grad.clone()
                opt.zero_grad(set_to_none=True)
          # Stage-tap probe update: MUST run before any extra model() call
          # (the pair block below re-fires the hooks on pair volumes).
          # Features are detached; only the probe heads receive gradient.
          # Guarded on `not _gc_defer`: a deferred micro-step did no forward
          # here at all, so tap_feats/_w/_y would be stale leftovers from an
          # earlier iteration - the gradcache window processes stage-taps
          # itself, once per study, from inside pass C's real forward.
          if not _gc_defer and tap_probes is not None and tap_feats:
              Bv, Sv, Zv = b["vol"].shape[:3]
              _msk = b["mask"].to(device).float()
              _yv = y.float()
              _wp = _w.detach().float()
              _tl = 0.0
              for si in stage_taps:
                  _f = tap_feats[si].reshape(Bv, Sv, Zv, -1).mean(2)
                  _f = ((_f * _msk.unsqueeze(-1)).sum(1)
                        / _msk.sum(1, keepdim=True).clamp(min=1))
                  _z = tap_probes[f"s{si}"](_f)
                  _bl = torch.nn.functional.binary_cross_entropy_with_logits(
                      _z, _yv, reduction="none")
                  _tl = _tl + (_bl * _wp).mean()
              tap_feats.clear()
              tap_opt.zero_grad(set_to_none=True)
              _tl.backward()
              tap_opt.step()
          if pair_iter is not None and i % args.pair_every == 0:
              # Longitudinal pair penalty, its own backward so the main loss
              # bookkeeping (tot/n, gold_train_w, logging) is untouched. Both
              # forwards live in the same graph: the gradient pushes the
              # earlier scan's OA logits down AND the later scan's up.
              pb = next(pair_iter)
              with torch.autocast("cuda", dtype=amp_dtype):
                  za = model(pb["a_vol"].to(device), pb["a_st"].to(device),
                             pb["a_mask"].to(device), pb["a_seq"].to(device))[0]
                  zb = model(pb["b_vol"].to(device), pb["b_st"].to(device),
                             pb["b_mask"].to(device), pb["b_seq"].to(device))[0]
              _pc = pb["cols"][0].to(device)
              pen = F.softplus(za.float()[:, _pc]
                               - zb.float()[:, _pc]).mean()
              scaler.scale(args.pair_rank * pen / args.accum).backward()
              if i % 200 == 0:
                  print(f"   [pair] penalty {pen.item():.4f}", flush=True)
          if bilat_iter is not None and i % args.bilat_every == 0:
              # Bilateral L/R ranking penalty (--bilat-rank). Both knees are
              # forwarded in THIS micro-step, so the accumulated gradient is
              # consumed by this window's optimizer step (both knees in the
              # same optimization step); no labels ride these forwards, so
              # the challenge loss is never double-counted; /accum matches
              # the main loss scaling.
              #
              # MEMORY-CHUNKED EXACT GRADIENTS: one L40S cannot hold two
              # study graphs at once (measured OOM, smoke job 59596310), so
              # the joint-graph backward is split per side: grad wrt each
              # knee's logits is taken with the PARTNER's logit values from
              # this same micro-step held constant (detached) - identical
              # to the joint gradient side-by-side (d pen/d z_side depends
              # on the partner only through its VALUES), evaluated on two
              # dropout draws. The partner is NOT a stale bank (the killed
              # fixed-bank failure mode): every application forwards both
              # knees fresh, so the pair genuinely reorders.
              bb = next(bilat_iter)
              _bcols = bb["cols"][0].to(device)
              _bdirs = bb["dirs"][0].to(device)
              _bstr = bb["strata"][0].to(device)
              _lin = (bb["l_vol"].to(device), bb["l_st"].to(device),
                      bb["l_mask"].to(device), bb["l_seq"].to(device))
              _rin = (bb["r_vol"].to(device), bb["r_st"].to(device),
                      bb["r_mask"].to(device), bb["r_seq"].to(device))
              with torch.no_grad(), \
                      torch.autocast("cuda", dtype=amp_dtype):
                  _zr0 = model(*_rin)[0].float()[0]      # values only
              with torch.autocast("cuda", dtype=amp_dtype):
                  _zl = model(*_lin)[0]                  # left graph live
              _penA, _bmA, _bc = bilat_pair_penalty(
                  _zl.float()[0], _zr0, _bcols, _bdirs, _bstr)
              _zl0 = _zl.float()[0].detach()
              scaler.scale(args.bilat_rank * _penA / args.accum).backward()
              with torch.autocast("cuda", dtype=amp_dtype):
                  _zr = model(*_rin)[0]                  # right graph live
              _penB, _bmB, _ = bilat_pair_penalty(
                  _zl0, _zr.float()[0], _bcols, _bdirs, _bstr)
              scaler.scale(args.bilat_rank * _penB / args.accum).backward()
              bilat_apps += 1
              _bpen = (float(_penA.detach()) + float(_penB.detach())) / 2
              bilat_pen_sum += _bpen
              for _s in BILAT_STRATA:
                  if _s in _bmA or _s in _bmB:
                      bilat_mass[_s] += (_bmA.get(_s, 0.0)
                                         + _bmB.get(_s, 0.0)) / 2
              for _s in _bc:
                  bilat_terms[_s] += _bc[_s]
              if i % 200 == 0:
                  print(f"   [bilat] penalty {_bpen:.4f}", flush=True)
          if auxpool_iter is not None and i % args.aux_pool_every == 0:
              # --aux-pool-*: one auxiliary-pool study, teacher soft-BCE over
              # all 12 cells at weight --aux-pool-w, /accum like the main
              # loss, its own backward into the SAME optimizer step (same
              # pattern as the pair/bilat blocks above; main-loss bookkeeping
              # untouched). Mass counters record the realised loss-weight
              # shares (main cells by their live weight _w; RSNA vs pool rows).
              ab = next(auxpool_iter)
              _ay = ab["y"].to(device, non_blocking=True)
              _aw = lw * ab["fw"].to(device) if "fw" in ab else lw
              with torch.autocast("cuda", dtype=amp_dtype):
                  _az = model(ab["vol"].to(device, non_blocking=True),
                              ab["stype"].to(device), ab["mask"].to(device),
                              ab["seq"].to(device))[0]
                  _al = soft_bce(_az, _ay, weight=_aw)
              scaler.scale(args.aux_pool_w * _al / args.accum).backward()
              auxpool_draws += 1
              auxpool_loss_sum += float(_al.detach())
              auxpool_mass += args.aux_pool_w * float(
                  (_aw * torch.ones_like(_ay)).sum())
              _mm = float((_w * torch.ones_like(y)).sum())
              auxpool_main_mass["pool" if b["study"][0] in cotrain
                                else "rsna"] += _mm
              if i % 1000 == 0:
                  _tot = (auxpool_mass + auxpool_main_mass["rsna"]
                          + auxpool_main_mass["pool"])
                  print(f"   [auxpool] ep{ep} mb{i}: draws {auxpool_draws} "
                        f"mean aux bce {auxpool_loss_sum / auxpool_draws:.4f}"
                        f" mass share rsna/pool/aux "
                        f"{auxpool_main_mass['rsna'] / _tot:.3f}/"
                        f"{auxpool_main_mass['pool'] / _tot:.3f}/"
                        f"{auxpool_mass / _tot:.3f}", flush=True)
          if (i + 1) % args.accum == 0:
              if bagauc_active and bagauc_buf:
                  # Deferred cross-study backward: the ONE point where every
                  # micro-step's retained graph in this window is used a
                  # second time, then freed (retain_graph defaults False).
                  # Not divided by accum - this is already a single
                  # window-level aggregate (mean over pairs, mean over 12
                  # columns), not a per-study term like the BCE above.
                  _bl, _bc = _bagauc_window_loss(bagauc_buf)
                  if _bl is not None:
                      scaler.scale(args.bagauc_w * _bl).backward()
                  bagauc_pair_counts += _bc
                  bagauc_n_windows += 1
                  bagauc_buf = []
              if bagauc_active and args.bagauc_gradcache and bagauc_gc_buf:
                  # GradCache two-pass window processing (see
                  # _bagauc_gradcache_window above) - the OOM fix. bagauc_buf
                  # above stays permanently empty in this mode (never
                  # appended to), so its block just did nothing; this is the
                  # actual window backward for --bagauc-gradcache runs. Does
                  # its own N per-study backward() calls internally, so
                  # nothing further is needed here before unscale/clip/step.
                  _is_first_win = (bagauc_n_windows == 0)
                  _gc_loss_sum, _gc_n, _bc = _bagauc_gradcache_window(
                      bagauc_gc_buf, ep, i, _is_first_win)
                  tot += _gc_loss_sum
                  n += _gc_n
                  bagauc_pair_counts += _bc
                  bagauc_n_windows += 1
                  bagauc_gc_buf = []
              if sam_state is not None:
                  # Hand the manually-accumulated SAM gradients to the
                  # optimizer path; from here on the step is a plain step.
                  with torch.no_grad():
                      for p, _g in sam_state["acc"].items():
                          p.grad = _g
                  sam_state["acc"].clear()
              # ---- grad-probe: fire BEFORE this genuine step --------------
              # The real accumulated gradients are already in .grad here, so
              # they are stashed and restored around the probe backwards. The
              # step that follows is bit-identical to an unprobed run's.
              if _gp_fire and _gp_step_i in _gp_fire and _gp_g is None:
                  for _nm, _v in (("bagauc_w", args.bagauc_w), ("dann", args.dann),
                                  ("aucm", args.aucm), ("text_w", args.text_w),
                                  ("sam_rho", getattr(args, "sam_rho", 0.0))):
                      if float(_v or 0) != 0:
                          raise SystemExit(f"--grad-probe: {_nm}={_v} on; probe "
                                           f"replicates soft_bce only")
                  _pp = [p for _, p in model.named_parameters() if p.requires_grad]
                  _stash = [None if p.grad is None else p.grad.detach().clone()
                            for p in _pp]
                  _gp_g = {f: [] for f in list(FINDINGS) + ["__joint__"]}
                  for _pb in _gp_batches:
                      for _tag in list(FINDINGS) + ["__joint__"]:
                          _w1 = torch.zeros(len(FINDINGS), device=device)
                          if _tag == "__joint__":
                              _w1[:] = 1.0
                          else:
                              _w1[FINDINGS.index(_tag)] = 1.0
                          opt.zero_grad(set_to_none=True)
                          with torch.autocast("cuda", dtype=amp_dtype,
                                              enabled=True):
                              _lg = model(_pb["vol"].to(device),
                                          _pb["stype"].to(device),
                                          _pb["mask"].to(device),
                                          _pb["seq"].to(device))[0]
                              if "lr_interval" in _pb:
                                  _ls = batch_label_bce(
                                      _lg, _pb["y"].to(device), _pb,
                                      weight=_w1)
                              else:
                                  _ls = soft_bce(
                                      _lg, _pb["y"].to(device), weight=_w1)
                          _ls.backward()
                          _gp_g[_tag].append(torch.cat(
                              [(p.grad if p.grad is not None
                                else torch.zeros_like(p)).detach().reshape(-1)
                               .float().cpu() for p in _pp]))
                  # restore the REAL accumulated gradients, exactly as found
                  opt.zero_grad(set_to_none=True)
                  for _p, _g in zip(_pp, _stash):
                      _p.grad = _g
                  _gp_before = torch.cat([p.detach().reshape(-1).float().cpu()
                                          for p in _pp])
                  _gp_target = _gp_fire[_gp_step_i]
                  _gp_armed = True
                  print(f"[gradprobe] target {_gp_target} (ep{ep} step "
                        f"{_gp_step_i}/{_gp_nsteps}): grads collected, "
                        f"measuring the next step", flush=True)
              scaler.unscale_(opt)
              # The pre-clip norm was computed and DISCARDED for the whole
              # project's history. If it exceeds 1.0 on most steps the
              # effective update is lr x unit-direction and OneCycle is not
              # the schedule actually running - measure before believing any
              # LR/schedule arm. Print-only; the update is unchanged.
              if getattr(args, "clip_groups", False):
                  _cth = args.clip if args.clip > 0 else float("inf")
                  _gnb = torch.nn.utils.clip_grad_norm_(
                      [p for n_, p in model.named_parameters()
                       if n_.startswith("backbone.")], _cth)
                  _gnr = torch.nn.utils.clip_grad_norm_(
                      [p for n_, p in model.named_parameters()
                       if not n_.startswith("backbone.")], _cth)
                  _gn = torch.sqrt(_gnb.float() ** 2 + _gnr.float() ** 2)
                  if len(_clip_norms) % 200 == 0:
                      print(f"[clip-groups] bb {float(_gnb):.3g} rest "
                            f"{float(_gnr):.3g}", flush=True)
              else:
                  _gn = torch.nn.utils.clip_grad_norm_(
                      model.parameters(),
                      args.clip if args.clip > 0 else float("inf"))
              _clip_norms.append(float(_gn))
              scaler.step(opt)
              if dann_opt is not None:
                  scaler.step(dann_opt)      # before update(), one update for both
              if aucm_opt is not None:
                  scaler.step(aucm_opt)
              scaler.update()
              # ---- grad-probe: the ACTUAL step just happened --------------
              _gp_step_i += 1
              if _gp_armed:
                  _pp = [p for _, p in model.named_parameters() if p.requires_grad]
                  _after = torch.cat([p.detach().reshape(-1).float().cpu()
                                      for p in _pp])
                  _dth = _after - _gp_before
                  if float(_dth.norm()) == 0.0:
                      # AMP skipped this step on a non-finite gradient; leave the
                      # probe armed and measure the next one instead of writing
                      # a dtheta of exactly zero.
                      print(f"[gradprobe] {_gp_target}: step skipped by "
                            f"AMP, staying armed", flush=True)
                  else:
                      _sz = [p.numel() for p in _pp]
                      _tm = torch.cat([torch.full((s,), bool(m))
                                       for s, m in zip(_sz, _gp_trunk)]).bool()
                      # AdamW state AFTER the step, for the ladder
                      _m = torch.cat([opt.state[p]["exp_avg"].detach().reshape(-1)
                                      .float().cpu() if p in opt.state and
                                      "exp_avg" in opt.state[p]
                                      else torch.zeros(p.numel()) for p in _pp])
                      _v = torch.cat([opt.state[p]["exp_avg_sq"].detach().reshape(-1)
                                      .float().cpu() if p in opt.state and
                                      "exp_avg_sq" in opt.state[p]
                                      else torch.zeros(p.numel()) for p in _pp])
                      _P = 1.0 / (_v.sqrt() + 1e-8)
                      _dad = _m * _P
                      _out = {"target": float(_gp_target), "epoch": int(ep),
                              "opt_step": int(_gp_step_i - 1),
                              "n_opt_steps": int(_gp_nsteps),
                              "n_batches": len(_gp_batches),
                              "dtheta_norm": float(_dth.norm()), "findings": {}}
                      _gj = torch.stack(_gp_g["__joint__"]).mean(0)
                      for _f in FINDINGS:
                          _rec = {}
                          for _scope, _msk in (("trunk", _tm),
                                               ("full", torch.ones_like(_tm))):
                              _d = _dth[_msk]
                              _dn = float(_d.norm()) + 1e-30
                              _C, _dL = [], []
                              for _gv in _gp_g[_f]:
                                  _gs = _gv[_msk]
                                  _dot = float(torch.dot(_gs, _d))
                                  _dL.append(_dot)
                                  _C.append(-_dot / (float(_gs.norm()) * _dn + 1e-30))
                              _gb = torch.stack(_gp_g[_f]).mean(0)[_msk]
                              _gjm = _gj[_msk]
                              _rec[_scope] = dict(
                                  C_mean=float(np.mean(_C)),
                                  C_all=[float(x) for x in _C],
                                  n_adverse=int(sum(1 for x in _C if x < 0)),
                                  dL_mean=float(np.mean(_dL)),
                                  L1_raw=float(torch.dot(_gb, _gjm)),
                                  L2_precond=float(torch.dot(_gb, _P[_msk] * _gjm)),
                                  L3_adam=float(torch.dot(_gb, _dad[_msk])),
                                  L4_actual=float(torch.dot(_gb, _d)))
                          _out["findings"][_f] = _rec
                      _tg = str(_gp_target).replace(".", "_")
                      _fp = os.path.join(args.grad_probe_out, f"probe_{_tg}.json")
                      json.dump(_out, open(_fp, "w"), indent=1)
                      torch.save({"opt": opt.state_dict(), "epoch": ep,
                                  "target": float(_gp_target)},
                                 os.path.join(args.grad_probe_out,
                                              f"optstate_{_tg}.pt"))
                      print(f"[gradprobe] {_gp_target}: wrote {_fp} "
                            f"(|dtheta|={float(_dth.norm()):.4e})", flush=True)
                      _gp_g, _gp_armed = None, False
              opt.zero_grad(set_to_none=True)
              if dann_opt is not None:
                  dann_opt.zero_grad(set_to_none=True)
              if aucm_opt is not None:
                  aucm_opt.zero_grad(set_to_none=True)
                  with torch.no_grad():
                      # dual ASCENT, then project to alpha >= 0
                      aucm_alpha.add_(args.aucm_lr * aucm_agrad / args.accum)
                      aucm_alpha.clamp_(min=0)
                      aucm_agrad.zero_()
              if ema is not None:
                  ema.update(model)      # after the step, not per micro-batch
                  for _x in ema_extra:
                      _x["ema"].update(model)
              _opt_steps += 1
              if _const_lr:
                  pass          # LR is fixed; the schedule is never stepped
              elif sched.last_epoch < min(steps - 1, _tail_hold):
                  sched.step()
              elif _tail_hold < steps and not _tail_announced:
                  _tail_announced = True
                  print(f"[lr-tail] hold begins at step {sched.last_epoch}/"
                        f"{steps}; lr held at "
                        f"{['%.3e' % g['lr'] for g in opt.param_groups]}",
                        flush=True)
              if _frz_ref and ep == 0 and i in (args.accum * 2 - 1,
                                                args.accum * 8 - 1):
                  # Fail in the first minute, not the first hour.
                  _frz_check(f"ep0 micro-batch {i}")
          if loss is not None:
              # None only on a deferred (_gc_defer) micro-step - that
              # study's contribution to tot/n is added in bulk, at the
              # window boundary above, by _bagauc_gradcache_window's return.
              tot += loss.item() * y.size(0)
              n += y.size(0)
          if i % 50 == 0:
              print(f"  ep{ep} {i}/{len(tr_dl)} loss {tot/max(n,1):.4f} "
                    f"{time.time()-t0:.0f}s", flush=True)

          # Sub-epoch EMA snapshot. Deliberately UNCONDITIONAL and unscored:
          # running the val loader mid-epoch would cost real training time,
          # and these files exist to be re-ranked offline against fixed
          # predictions, where selection carries no seed noise. fold_auc /
          # gold_auc are therefore absent by design, not missing by accident
          # - the name is fold{N}_sub{ep}_{pct}.pt, which no existing
          # consumer glob matches (checked: oai_free_registry.py takes
          # fold*_ema*.pt).
          if i in _sub_at:
              _f = _sub_at[i]
              _stem = f"fold{args.fold}_sub{ep}_{int(round(_f*100)):02d}"
              _meta = {"args": vars(args), "epoch": ep,
                       "epoch_float": ep + _f, "subepoch": True,
                       "subepoch_frac": _f, "ema": True, "unscored": True,
                       "prob_table": prob_table}
              torch.save({"model": ema.state_dict(model), **_meta},
                         os.path.join(args.out, _stem + ".pt"))
              # Extra EMA shadows at the SAME sub-epoch point. The decay
              # sweep has never been evaluated in this window:
              # --ema-extra-start defaults to 4 and its help says it skips
              # "epochs no selector has ever picked (selection lands
              # ep7-10)" - but the Synovitis sub-roster selects at ep0-1,
              # exactly the range that reasoning discards. It matters more
              # here than there: the ramp caps effective decay at ~0.9983,
              # so in the mature window 0.999/0.9999/1.0 are the same
              # shadow, while early the ramp is still active and the
              # decays actually separate. Saved unscored, so this costs
              # disk and no wall-clock - set --ema-extra-start above
              # --epochs to suppress the per-epoch scoring passes entirely.
              for _x in ema_extra:
                  torch.save({"model": _x["ema"].state_dict(model),
                              "ema_decay": _x["decay"], **_meta},
                             os.path.join(args.out,
                                          f"{_stem}_d{_x['tag']}.pt"))
              print(f"   saved sub-epoch EMA snapshot (ep{ep}+{_f:.2f}, "
                    f"batch {i}/{len(tr_dl)}, +{len(ema_extra)} shadow(s))",
                    flush=True)

        if bagauc_active and bagauc_buf:
            # Leftover partial window at epoch end (len(tr_dl) % accum != 0).
            # Flushed here rather than carried into next epoch's data loader
            # so no live graph survives an epoch boundary - the leftover
            # GRADIENT is allowed to carry over (matching the pre-existing
            # BCE behaviour: opt.zero_grad only fires on a full accum step),
            # but the retained graph itself must not.
            _bl, _bc = _bagauc_window_loss(bagauc_buf)
            if _bl is not None:
                scaler.scale(args.bagauc_w * _bl).backward()
            bagauc_pair_counts += _bc
            bagauc_n_windows += 1
            bagauc_buf = []
        if bagauc_active and args.bagauc_gradcache and bagauc_gc_buf:
            # Leftover partial window at epoch end, gradcache mode. `i`
            # still holds its value from the last for-loop iteration (the
            # global step of the last buffered study), matching what
            # win_start_i's derivation inside _bagauc_gradcache_window
            # expects.
            _is_first_win = (bagauc_n_windows == 0)
            _gc_loss_sum, _gc_n, _bc = _bagauc_gradcache_window(
                bagauc_gc_buf, ep, i, _is_first_win)
            tot += _gc_loss_sum
            n += _gc_n
            bagauc_pair_counts += _bc
            bagauc_n_windows += 1
            bagauc_gc_buf = []
        if bagauc_active:
            _tot_pairs = int(bagauc_pair_counts.sum())
            _mean_pw = _tot_pairs / max(bagauc_n_windows, 1)
            print(f"[bagauc] ep {ep}: {bagauc_n_windows} windows, "
                  f"{_tot_pairs} total cross-bag pairs "
                  f"({_mean_pw:.1f}/window, max possible per-finding "
                  f"C(accum,2)={args.accum * (args.accum - 1) // 2}); "
                  "by-finding pairs/window: " + " ".join(
                      f"{f.split()[0][:4]}:{c/max(bagauc_n_windows,1):.1f}"
                      for f, c in zip(FINDINGS, bagauc_pair_counts.tolist())),
                  flush=True)

        # ---- --ret-anchor: per-epoch audit + end-of-anchor-epoch build ----
        # (scratch/retanchor/PREREG.md). The audit line exists so bar reads
        # and debugging never need checkpoint surgery; the same execution-
        # audit discipline as [moaksaux]/logitmse - a flagged arm whose term
        # reached zero batches after arming is not running the intervention
        # it claims to, and that is fatal, not cosmetic.
        if args.ret_anchor_w > 0:
            print(f"   [retanchor] ep{ep} audit: {retanchor_steps} batches "
                  f"supervised, mean anchor-BCE "
                  f"{retanchor_sum / max(retanchor_steps, 1):.4f}, "
                  f"w {args.ret_anchor_w}", flush=True)
            if ep <= args.ret_anchor_epoch:
                assert retanchor_steps == 0, \
                    f"retanchor term fired in ep{ep} BEFORE the anchor " \
                    f"epoch {args.ret_anchor_epoch} completed"
            elif retanchor_steps == 0:
                raise SystemExit(
                    f"--ret-anchor-w reached ZERO batches in epoch {ep} "
                    f"despite an armed anchor - refusing to continue as a "
                    f"silent no-op arm.")
            if ep == args.ret_anchor_epoch:
                # END of the anchor epoch: this point is after the batch
                # loop, i.e. after the epoch's last optimizer step and EMA
                # update, and before the epoch-end evals. Save the EMA
                # snapshot (standard checkpoint dict conventions), then arm
                # a frozen in-memory copy for every subsequent epoch.
                # deepcopy is safe here: the cell-stage hook (the only
                # closure-over-self hook reachable at train time) is
                # excluded up front at flag validation.
                import copy as _copy
                _ra_path = os.path.join(args.out,
                                        f"fold{args.fold}_anchor_ema.pt")
                torch.save({"model": ema.state_dict(model),
                            "args": vars(args), "epoch": ep, "ema": True,
                            "ret_anchor": True, "prob_table": prob_table},
                           _ra_path)
                # forward() stashes NON-LEAF graph tensors on the module
                # (_study_vec every forward; _trunk_rep under gram-escrow;
                # _cell_feat under cell-attn), and torch refuses to deepcopy
                # non-leaf tensors (smoke job 59596445 failed exactly here).
                # They are per-batch scratch, dead at the epoch boundary, and
                # the next forward recreates them - clearing is safe. Only
                # ever executed inside this flagged block, so flag-off
                # parity is untouched.
                for _sn in ("_study_vec", "_trunk_rep", "_cell_feat",
                            "_patch_feat", "_seg_logits"):
                    if getattr(model, _sn, None) is not None:
                        setattr(model, _sn, None)
                ret_anchor_model = _copy.deepcopy(model)
                ret_anchor_model.load_state_dict(ema.state_dict(model))
                ret_anchor_model.eval()
                for _p in ret_anchor_model.parameters():
                    _p.requires_grad_(False)
                print(f"   [retanchor] ep{ep}: anchor EMA saved -> "
                      f"{_ra_path}; frozen in-memory anchor armed from "
                      f"ep{ep + 1}", flush=True)

        if _clip_norms:
            _cn = sorted(_clip_norms)
            _thr = args.clip if args.clip > 0 else 1.0
            _bind = sum(1 for v in _cn if v > _thr) / len(_cn)
            _lbl = "binding" if args.clip > 0 else "ref>1.0"
            print(f"[clip] ep {ep}: median {_cn[len(_cn)//2]:.3f}  "
                  f"p90 {_cn[int(len(_cn)*0.9)]:.3f}  "
                  f"{_lbl} {100*_bind:.1f}%  (n={len(_cn)})", flush=True)

        _frz_check(f"end of ep{ep}")
        if dismix is not None:   # dose print at the END of the epoch
            print(f"[dismix] epoch {ep} end: {dismix.summary(FINDINGS)}", flush=True)
        Pv, Yv, Uv = evaluate(model, va_dl, device, amp_dtype)
        # Selection runs on the ceiling-normalised mean; the raw value is
        # logged too because every earlier run reported it.
        va_auc = mean_soft_auc(Pv, Yv, normalise=True, cols=sel_cols)
        va_raw = mean_soft_auc(Pv, Yv)
        va_hard, _ = mean_auc(Pv, Yv)           # reported for continuity
        v_loss, v_base, v_floor = bce_and_floors(Pv, Yv)
        if gold_dl is None:
            g_auc = float("nan")
            g_each = np.full(len(FINDINGS), np.nan, dtype=np.float32)
        else:
            Pg, Yg, _ = evaluate(model, gold_dl, device, amp_dtype)
            g_auc, g_each = mean_auc(Pg, Yg)
        # The flat-attention branch is a ZERO-init gated residual, so at step 0
        # its projection gets no gradient at all - only the gate does. That is
        # the same shape as the seq_emb failure, where a feature sat inert
        # through a 5.5-hour run and the arm could not answer its own question.
        # Printing the gate makes "SGD tried it and declined" distinguishable
        # from "the branch was never engaged", per epoch, for free.
        # BOTH gates, not just flat. When --cell-attn was added this logging was
        # not extended, so cellattn/cellfine ran with no gate trace at all - the
        # exact diagnostic built to catch the flat_attn collapse, absent from the
        # arms it mattered most for. Recovered from checkpoints that time.
        _gs = []
        for _n in ("flat_gate", "cell_gate"):
            _g = getattr(getattr(model, "study_enc", None), _n, None)
            if _g is not None:
                _gs.append(f"{_n.replace('_gate','')}-gate {_g.detach().abs().item():.4f}")
        _fg = ("  " + "  ".join(_gs)) if _gs else ""
        # EXECUTION AUDIT for --silent-mute-from. The day-3 --mute-after arm is
        # only citable because its zeroing was audited across head, bias, aux
        # head and routing query rather than assumed from the flag. Same rule
        # here: report the cells actually muted this epoch, per finding, so
        # "the flag was set" is never mistaken for "the mask fired".
        if rcw_col is not None:
            print(f"   [rcw] ep{ep} audit: {rcw_cells_zeroed} report cells "
                  f"zero-weighted (findings "
                  f"{[k for k, v in zip(FINDINGS, rcw_col) if v == 0]})",
                  flush=True)
        if bool(silmute_grade):
            _sm_seen = max(silmute_studies_seen, 1)
            print(f"   [silmute] ep{ep} audit: muted "
                  f"{int(silmute_cell_counts.sum())} cells over "
                  f"{silmute_studies_seen} studies "
                  f"({silmute_cell_counts.sum() / (_sm_seen * len(FINDINGS)):.3f} "
                  f"of cells); UNRESOLVED studies {silmute_unresolved} "
                  f"(MUST be 0); per-finding "
                  + " ".join(f"{_f[:4]}:{int(_c)}"
                             for _f, _c in zip(FINDINGS, silmute_cell_counts)),
                  flush=True)
        # EXECUTION AUDIT for --distill-logit-mse. The startup check proved the
        # row set is non-empty; this proves the term actually fired inside the
        # loop, and a zero count is fatal rather than cosmetic - a flagged arm
        # that silently trained plain soft-BCE would be adjudicated as a NULL
        # for the mechanism instead of as a broken run.
        if args.distill_logit_mse > 0:
            print(f"   [logitmse] ep{ep} audit: applied on {lmse_steps} "
                  f"micro-steps / {lmse_cells} cells; mean MSE "
                  f"{lmse_sum / max(lmse_steps, 1):.6g}; w "
                  f"{args.distill_logit_mse}", flush=True)
            if lmse_cells == 0:
                raise SystemExit(
                    f"--distill-logit-mse: the term reached ZERO cells in "
                    f"epoch {ep} despite {len(distill_uids)} eligible training "
                    f"studies - the arm is not running the intervention it "
                    f"claims to. Refusing to continue.")
        # EXECUTION AUDIT for --aux-seg-w, same rule: a nonzero coverage
        # count at startup does not prove the term fired inside the loop.
        if args.aux_seg_w > 0:
            print(f"   [auxseg] ep{ep} audit: applied on {auxseg_steps} "
                  f"micro-steps / {auxseg_series} (series,Z-window) rows; "
                  f"mean CE+Dice {auxseg_sum / max(auxseg_steps, 1):.6g}; w "
                  f"{args.aux_seg_w}", flush=True)
            if auxseg_steps == 0:
                raise SystemExit(
                    f"--aux-seg-w: the term reached ZERO micro-steps in "
                    f"epoch {ep} despite nonzero --aux-seg-root coverage at "
                    f"startup - the arm is not running the intervention it "
                    f"claims to. Refusing to continue.")
        # EXECUTION AUDIT for --gram-escrow, same rule as silmute's: report
        # what the constraint ACTUALLY did, so "the flag was set" is never
        # mistaken for "the penalty bound". mean L_gram is the descriptive
        # did-it-bind read (PREREG 7.6); MISSING must be 0 or the anchor and
        # the train pool disagree and the arm is invalid.
        if gramesc_on:
            print(f"   [gramesc] ep{ep} audit: penalised {gram_n} micro-steps; "
                  f"mean L_gram {gram_sum / max(gram_n, 1):.6g}; "
                  f"MISSING anchors {gram_missing} (MUST be 0); "
                  f"lambda {args.gram_escrow_w}", flush=True)
        # EXECUTION AUDIT for --moaks-aux, same rule as logitmse's: the term
        # must have reached real cells this epoch or the arm is not running
        # the intervention it claims to, and that is fatal, not cosmetic.
        if moaks_names:
            print(f"   [moaksaux] ep{ep} audit: supervised {moaks_cells} "
                  f"cells over {moaks_steps} micro-steps; mean masked BCE "
                  f"{moaks_sum / max(moaks_steps, 1):.4f}; "
                  f"w {args.moaks_aux_w}", flush=True)
            if moaks_cells == 0:
                raise SystemExit(
                    f"--moaks-aux reached ZERO cells in epoch {ep} despite "
                    f"loaded targets - refusing to continue as a silent "
                    f"no-op arm.")
        # EXECUTION AUDIT + frozen falsification read for --bilat-rank
        # (scratch/oaibilat/PREREG.md): per epoch, the ranking-loss MASS
        # split by discordance stratum, so ">= 60% of ranking-loss mass in
        # STRONG-discordant pairs" is asserted straight from the log. TIE
        # comp-terms are excluded at build time (count printed at startup
        # and repeated here), so their mass is structurally zero.
        if bilat_iter is not None:
            _btm = bilat_mass["STRONG"] + bilat_mass["MILD"]
            _bfr = {s: (bilat_mass[s] / _btm if _btm > 0 else float("nan"))
                    for s in BILAT_STRATA}
            print(f"   [bilat] ep{ep} audit: {bilat_apps} pair applications, "
                  f"mean penalty {bilat_pen_sum / max(bilat_apps, 1):.4f}; "
                  f"loss mass STRONG {bilat_mass['STRONG']:.4g} "
                  f"({_bfr['STRONG']:.1%}) MILD {bilat_mass['MILD']:.4g} "
                  f"({_bfr['MILD']:.1%}); comp-terms STRONG "
                  f"{bilat_terms['STRONG']} MILD {bilat_terms['MILD']} "
                  f"TIE-excluded-at-build {bilat_tie_comps}; "
                  f"w {args.bilat_rank} "
                  f"[falsification: STRONG mass >= 60%]", flush=True)
            if bilat_apps == 0:
                raise SystemExit(
                    f"--bilat-rank reached ZERO pair applications in epoch "
                    f"{ep} despite loaded pairs - refusing to continue as a "
                    f"silent no-op arm.")
        if auxpool_iter is not None:
            _tot = (auxpool_mass + auxpool_main_mass["rsna"]
                    + auxpool_main_mass["pool"])
            print(f"   [auxpool] ep{ep} audit (cumulative): {auxpool_draws} "
                  f"aux draws, mean aux bce "
                  f"{auxpool_loss_sum / max(auxpool_draws, 1):.4f}; loss "
                  f"mass rsna {auxpool_main_mass['rsna']:.6g} pool "
                  f"{auxpool_main_mass['pool']:.6g} aux {auxpool_mass:.6g} = "
                  f"shares {auxpool_main_mass['rsna'] / max(_tot, 1e-12):.4f}/"
                  f"{auxpool_main_mass['pool'] / max(_tot, 1e-12):.4f}/"
                  f"{auxpool_mass / max(_tot, 1e-12):.4f} "
                  f"[prereg: .455/.401/.144]", flush=True)
            if auxpool_draws == 0:
                raise SystemExit("--aux-pool-teacher reached ZERO aux draws; "
                                 "refusing to continue as a silent no-op arm")
        if _vis_tab is not None:
            print(f"[aux-vis] ep {ep} kept-fraction "
                  f"{_vis_acc[0]/max(_vis_acc[1],1.0):.4f} "
                  f"({int(_vis_acc[0])}/{int(_vis_acc[1])} aux cells)", flush=True)
        if witness is not None:
            print(f"[witness] ep {ep}: L_witness mean "
                  f"{_wit_ep[1]/max(_wit_ep[0],1):.5f} over {_wit_ep[0]} "
                  f"batches, {_wit_ep[2]} gated cells, beta="
                  f"{args.witness_beta}", flush=True)
        if slicewit is not None:
            print(f"[slicewit] ep {ep}: cumulative L_slicewit mean "
                  f"{_sw_ct[1]/max(_sw_ct[0],1):.5f} over {_sw_ct[0]} batches, "
                  f"beta={args.slicewit_beta}", flush=True)
        print(f"[ep {ep}] loss {tot/max(n,1):.4f}  fold-softAUCn {va_auc:.4f} "
              f"(raw {va_raw:.4f})  "
              f"fold-AUC@.5 {va_hard:.4f}  {_gold_tag} {g_auc:.4f}  "
              f"| val-loss {v_loss:.4f} (base {v_base:.4f}, floor {v_floor:.4f})"
              f"{_fg}  ({time.time()-t0:.0f}s)", flush=True)
        if gold_dl is not None:
            print("   gold per-finding: " +
                  " ".join(f"{f.split()[0][:4]}:{a:.2f}" for f, a in zip(FINDINGS, g_each)),
                  flush=True)
        # ---- MOAKS aux learnability read (PREREG 4a) + OA-pred dump -------
        # Held-out-OAI knees, raw end-of-epoch weights (the EMA blocks below
        # are untouched to keep their cost and byte behaviour identical).
        # Per-column AUC against the binarised ordinal target (>= grade 2,
        # i.e. ladder prob > 0.5); block means printed so the gate (area
        # >= 0.70, depth >= 0.65) reads straight off the log. The per-study
        # challenge-OA-head probabilities are appended to a jsonl next to the
        # checkpoints together with the sidecar's floored-vs-rubric agreement
        # membership (amendment 1: descriptive dump only, no gate).
        if moaks_val_dl is not None:
            _MP, _MO, _MY, _MU = [], [], [], []
            model.eval()
            with torch.no_grad():
                for _mb in moaks_val_dl:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        _mo = model(_mb["vol"].to(device, non_blocking=True),
                                    _mb["stype"].to(device),
                                    _mb["mask"].to(device),
                                    _mb["seq"].to(device))
                    _MP.append(torch.sigmoid(_mo[-1].float()).cpu().numpy())
                    _MO.append(torch.sigmoid(_mo[0].float()).cpu().numpy())
                    _MY.append(_mb["y_aux"].numpy())
                    _MU += list(_mb["study"])
            _MP = np.concatenate(_MP)
            _MO = np.concatenate(_MO)
            _MY = np.concatenate(_MY)
            _m_aucs = np.full(len(moaks_names), np.nan)
            for _j in range(len(moaks_names)):
                _fin = np.isfinite(_MY[:, _j])
                if _fin.sum():
                    _m_aucs[_j] = auc_score(
                        (_MY[_fin, _j] > 0.5).astype(int), _MP[_fin, _j])
            _parts = []
            for _bn, _sl in zip(moaks_block_names, moaks_block_slices):
                _bv = _m_aucs[_sl]
                _bv = _bv[np.isfinite(_bv)]
                _parts.append(f"{_bn} {float(np.mean(_bv)):.3f}({len(_bv)})"
                              if len(_bv) else f"{_bn} nan(0)")
            print(f"   [moaks-val] ep{ep} held-out {len(_MU)} studies, "
                  f"block AUC (cols with both classes): " + "  ".join(_parts)
                  + "   [gate: area>=0.70 depth>=0.65]", flush=True)
            _oa_dump = os.path.join(args.out, "moaks_holdout_oa_preds.jsonl")
            with open(_oa_dump, "a") as _fh:
                for _k, _u in enumerate(_MU):
                    _mm = moaks_meta.get(_u, {})
                    _fh.write(json.dumps({
                        "ep": ep, "uid": _u,
                        "oa_probs": {"Medial OA": float(_MO[_k, 4]),
                                     "Lateral OA": float(_MO[_k, 5]),
                                     "PF OA": float(_MO[_k, 6])},
                        "oa_agree": _mm.get("oa_agree"),
                        "oa_floor_pos": _mm.get("oa_floor_pos"),
                        "oa_rubric_pos": _mm.get("oa_rubric_pos")}) + "\n")
            print(f"   [moaks-val] appended {len(_MU)} OA-head rows -> "
                  f"{_oa_dump}", flush=True)
        if sel_cols is not None and gold_dl is not None:
            # The headline GOLD-AUC above averages twelve columns, eleven of
            # which are untrained here, so it is meaningless for this arm. This
            # is the number the specialist is actually judged on.
            print("   SELECTED gold: " + "  ".join(
                f"{FINDINGS[j]} {g_each[j]:.4f}" for j in sel_cols), flush=True)

        # Unconditional early snapshot, for per-finding checkpoint selection.
        # This is NOT a curiosity: serving Synovitis from ep0 checkpoints while
        # the roster served everything else from ep6-10 was +0.003 LB
        # (0.946 -> 0.949, = +0.036 on Synovitis alone) on 2026-08-15. Findings
        # peak at different epochs and the ensemble was averaging members
        # trained past the peak. Every seat must therefore ship an early
        # checkpoint, or it cannot participate in that scheme.
        if ep == args.save_early_epoch:
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "fold_auc": va_auc, "gold_auc": g_auc, "epoch": ep,
                        "early": True, "prob_table": prob_table},
                       os.path.join(args.out, f"fold{args.fold}_early.pt"))
            print(f"   saved early snapshot (ep{ep}, gold {g_auc:.4f})", flush=True)
            # EMA version too. The raw-weights snapshot above was all that
            # existed, but EMA wins 8/8 and roughly HALVES seed variance
            # (0.0079 -> 0.0047), and the checkpoints that actually scored
            # 0.949 were EMA. A raw early snapshot is the weaker artifact.
            if ema is not None:
                torch.save({"model": ema.state_dict(model), "args": vars(args),
                            "fold_auc": va_auc, "gold_auc": g_auc, "epoch": ep,
                            "early": True, "ema": True,
                            "prob_table": prob_table},
                           os.path.join(args.out,
                                        f"fold{args.fold}_early_ema.pt"))
                print(f"   saved early EMA snapshot (ep{ep})", flush=True)

        if va_auc > best:
            best = va_auc
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "fold_auc": va_auc, "gold_auc": g_auc, "epoch": ep,
                        "prob_table": prob_table},
                       os.path.join(args.out, f"fold{args.fold}_best.pt"))
            # Save the val-fold predictions alongside. These are the
            # out-of-fold predictions for this fold's studies, and the union
            # over folds is what self-training consumes and what the extraction
            # referee scores against - neither is possible from checkpoints
            # alone without a second inference pass over every fold.
            np.savez_compressed(
                os.path.join(args.out, f"fold{args.fold}_oof.npz"),
                uids=np.array(Uv), preds=Pv.astype(np.float32),
                targets=Yv.astype(np.float32), epoch=ep, fold=args.fold)
            print(f"   saved (fold-AUC {va_auc:.4f}) + {len(Uv)} OOF preds",
                  flush=True)

        # The EMA weights are scored and tracked independently, and written to a
        # separate file. Nothing picks between the two here: that comparison is
        # what the next few epochs of logs are for, and choosing per-epoch on the
        # same validation set we already select epochs on would just add noise.
        if ema is not None:
            raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict(model))
            Pv_e, Yv_e, _ = evaluate(model, va_dl, device, amp_dtype)
            e_auc = mean_soft_auc(Pv_e, Yv_e, normalise=True, cols=sel_cols)
            if gold_dl is None:
                e_gold = float("nan")
            else:
                Pg_e, Yg_e, _ = evaluate(model, gold_dl, device, amp_dtype)
                e_gold, _ = mean_auc(Pg_e, Yg_e)
            print(f"   ema: fold-softAUC {e_auc:.4f} (raw {va_auc:.4f})  "
                  f"GOLD-AUC {e_gold:.4f} (raw {g_auc:.4f})", flush=True)
            if e_auc > best_ema:
                best_ema = e_auc
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "fold_auc": e_auc, "gold_auc": e_gold, "epoch": ep,
                            "ema": True, "prob_table": prob_table},
                           os.path.join(args.out, f"fold{args.fold}_ema.pt"))
                print(f"   saved ema (fold-AUC {e_auc:.4f})", flush=True)
            # SECOND selector, saved to a SEPARATE file: the 101-arm log audit
            # (2026-08-13) measured plain fold-AUC@.5 picking epochs worth
            # +0.0018 mean gold over softAUCn (paired SE 0.0010, half the
            # oracle gap) - the selection-metric twin of the rank-mean
            # combiner finding. Dual-save rather than a switch so every
            # historical family comparison on fold{N}_ema.pt stays valid;
            # rosters/rig may price the _a5 checkpoint independently.
            e_hard, _ = mean_auc(Pv_e, Yv_e)
            # PRINT-ONLY selector telemetry: nothing here picks anything. The
            # a5 audit settled soft-vs-hard but not WHERE to binarise, and the
            # threshold is not a free parameter - the grade ladder is
            # 0.056/0.358/0.708/0.931, so @.3 counts grade-1 as positive, @.5
            # splits grade-1 from grade-2, @.8 demands grade-3. Three different
            # rulers, one of which is shipped. Logging all three per epoch lets
            # the next log audit price the choice retrospectively across every
            # arm, the way the 101-arm audit priced soft-vs-hard, at zero cost.
            e_t3, _ = mean_auc(Pv_e, Yv_e, thresh=0.3)
            e_t8, _ = mean_auc(Pv_e, Yv_e, thresh=0.8)
            print(f"   ema-sel: @.3 {e_t3:.4f}  @.5 {e_hard:.4f}  "
                  f"@.8 {e_t8:.4f}  softn {e_auc:.4f}", flush=True)
            if e_hard > best_ema_a5:
                best_ema_a5 = e_hard
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "fold_auc": e_hard, "gold_auc": e_gold, "epoch": ep,
                            "ema": True, "selector": "auc5",
                            "prob_table": prob_table},
                           os.path.join(args.out, f"fold{args.fold}_ema_a5.pt"))
                print(f"   saved ema_a5 (fold-AUC@.5 {e_hard:.4f})", flush=True)
            if tap_probes is not None:
                torch.save({"probes": tap_probes.state_dict(),
                            "taps": stage_taps, "epoch": ep,
                            "args": vars(args)},
                           os.path.join(args.out,
                                        f"fold{args.fold}_stage_probes.pt"))
            # Extra decays. Selected on fold-AUC@.5 so the comparison is
            # against fold{N}_ema_a5.pt and varies ONLY the decay - pricing
            # these against fold{N}_ema.pt would confound window with selector.
            if ep >= args.ema_extra_start:
                for _x in ema_extra:
                    model.load_state_dict(_x["ema"].state_dict(model))
                    Pv_x, Yv_x, _ = evaluate(model, va_dl, device, amp_dtype)
                    x_hard, _ = mean_auc(Pv_x, Yv_x)
                    x_soft = mean_soft_auc(Pv_x, Yv_x, normalise=True,
                                           cols=sel_cols)
                    if gold_dl is None:
                        x_gold = float("nan")
                    else:
                        Pg_x, Yg_x, _ = evaluate(model, gold_dl, device, amp_dtype)
                        x_gold, _ = mean_auc(Pg_x, Yg_x)
                    print(f"   ema d{_x['decay']}: fold-AUC@.5 {x_hard:.4f} "
                          f"(a5 {e_hard:.4f})  fold-softAUC {x_soft:.4f}  "
                          f"GOLD-AUC {x_gold:.4f} (a5 {e_gold:.4f})", flush=True)
                    if x_hard > _x["best"]:
                        _x["best"] = x_hard
                        torch.save({"model": model.state_dict(),
                                    "args": vars(args), "fold_auc": x_hard,
                                    "gold_auc": x_gold, "epoch": ep,
                                    "ema": True, "selector": "auc5",
                                    "ema_decay_actual": _x["decay"],
                                    "prob_table": prob_table},
                                   os.path.join(
                                       args.out,
                                       f"fold{args.fold}_ema_d{_x['tag']}.pt"))
                        print(f"   saved ema_d{_x['tag']} "
                              f"(fold-AUC@.5 {x_hard:.4f})", flush=True)
            model.load_state_dict(raw_sd)

        if getattr(args, "fork_save_epoch", None) == ep + 1:
            _fp = os.path.join(args.out, f"fold{args.fold}_fork_ep{ep + 1}.pt")
            torch.save({
                "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "ema_shadow": ({k: v.cpu() for k, v in ema.shadow.items()}
                               if ema is not None else None),
                "ema_n": ema.n if ema is not None else None,
                "ema_extra": [{"shadow": {k: v.cpu() for k, v in
                                          _x["ema"].shadow.items()},
                               "n": _x["ema"].n} for _x in ema_extra],
                "opt_steps": _opt_steps, "epoch": ep, "args": vars(args),
                "rng_py": random.getstate(), "rng_np": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None)}, _fp)
            print(f"[fork-save] full training state after ep{ep} -> {_fp} "
                  f"(opt_steps={_opt_steps})", flush=True)
        if (getattr(args, "fork_resume", "") and ema is not None
                and ep == args.epochs - 1):
            torch.save({"model": ema.state_dict(model), "args": vars(args),
                        "epoch": ep, "ema": True, "terminal": True,
                        "prob_table": prob_table},
                       os.path.join(args.out,
                                    f"fold{args.fold}_fork_terminal_ema.pt"))
            print(f"[fork-terminal] saved terminal EMA after ep{ep}", flush=True)

        if _steps_done or (args.max_steps >= 0 and _opt_steps >= args.max_steps):
            print(f"[max-steps] halting after {_opt_steps} optimizer steps "
                  f"(epoch {ep}); one validation/save was written",
                  flush=True)
            break

        if args.stop_epoch and ep + 1 >= args.stop_epoch:
            print(f"[stop-epoch] halting after ep{ep} by request; LR "
                  f"schedule was built for {args.epochs} epochs so the "
                  f"trajectory so far is schedule-matched to full runs",
                  flush=True)
            break

    if pcol_mod is not None:
        torch.save({"proto": pcol_mod.proto.detach().cpu(), "w": pcol_mod.w.cpu(),
                    "steps": pcol_steps, "cells": pcol_cells,
                    "mean_loss": pcol_loss_sum / max(1, pcol_steps)},
                   os.path.join(args.out, "pcol_protos.pt"))
        print(f"[pcol] done: {pcol_steps} steps, {pcol_cells} graded cells, "
              f"mean aux loss {pcol_loss_sum / max(1, pcol_steps):.4f}; "
              f"prototypes saved", flush=True)
    print(f"[done] best fold-AUC {best:.4f}"
          + (f"  |  best ema {best_ema:.4f}" if ema is not None else ""), flush=True)


if __name__ == "__main__":
    main()
