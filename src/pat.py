"""PAT arm - patching-based counterfactual TRAINING (scratch/shortcutarms/PREREG.md §PAT).

Xie et al., ICML 2024 ("Counterfactual Reasoning for Multi-Label Image
Classification via Patching-Based Training", github.com/xiemk/MLC-PAT). In the
reference code (run_PAT-T.py) an image AND its n_grid x n_grid crops (default
n_grid=2) are forwarded independently; the per-class patch logits are combined by
a softmax over patches (`weighted_sum`: per-class softmax attention over the
patches, so the PIVOT patch - the one that holds the target - dominates) and the
loss is `criterion(logits_ori, y) + criterion(logits_pat, y)`, equal weights, same
image targets. Supervising the pivot-dominated patch aggregate with the image label
enhances the DIRECT effect of the target object: a patch cannot see a companion
object that lives in another patch.

Our adaptation (training-only; serve path byte-identical):
  * a "patch" is one contiguous SLAB of pat_grid equal slot ranges of ONE series
    (1-D analogue of the 2x2 grid: half extent along the slice axis, n_grid=2 -> 2
    slabs of 12 of the 24 slots; in sagittal series the slabs are the medial and
    lateral halves of the knee). Series are already encoded independently in the
    EV head, so the only within-study context a patch removes is the slice-
    transformer context from the OTHER slab of the same series, plus every other
    series.
  * the slab is re-encoded by the SAME slice transformer on its own slots (its
    self-attention cannot see the other slab), read by the SAME ev_head, pooled by
    EXACTLY the V1 pool, and its patch logit is ev_b + ep (a one-patch study:
    reliability weight 1).
  * the backbone slice features `f` are REUSED from the full forward (ConvNeXt
    sees only its own 3-slice input stack, so a slice's feature does not depend on
    which slab it is in). Deviation, declared: the slab-boundary slice keeps its
    one adjacent-slot pixel channel from across the boundary.
  * aggregation = per-finding softmax over the valid (study's present series x
    slabs) patches of the patch logits, times the patch logits ("self" form of the
    reference `weighted_sum`; the reference's 'cross' form takes attention and
    values from two different heads, which our single evidence head does not
    have). The pivot is therefore chosen by the model's OWN evidence, softly.
"""
import math

import torch
import torch.nn.functional as F


def ev_pool(model, e, Z):
    """e (N, Z, F) float raw per-slice evidence -> (N, F) per-series evidence.

    EXACTLY the V1 pool of KneeNet._evidence_forward (3-tap edge-replicated mean
    then per-finding LSE at ev_tau.clamp(0.2) minus log Z on the focal columns;
    softmax(ev_mix) mixture of mean / quantile-.8 / max on the diffuse columns).
    ev2 / syno_specialist / ev_adapter are refused in the KneeNet ctor under
    --pat, so there is nothing else to replicate (unit-tested against ep).
    """
    N, _, n_out = e.shape
    es = e.permute(0, 2, 1).reshape(N * n_out, 1, Z)
    es = F.avg_pool1d(F.pad(es, (1, 1), mode="replicate"), 3, stride=1)
    es = es.reshape(N, n_out, Z).permute(0, 2, 1)                 # (N,Z,F)
    tau = model.ev_tau.clamp(min=0.2)
    lse = tau * (torch.logsumexp(es / tau, dim=1) - math.log(float(Z)))
    di = model.ev_dif_idx
    ed = e[..., di]
    w = torch.softmax(model.ev_mix, dim=-1)
    mix = (w[:, 0] * ed.mean(1)
           + w[:, 1] * torch.quantile(ed, 0.8, dim=1)
           + w[:, 2] * ed.max(1).values)
    ep = lse.clone()
    ep[..., di] = mix
    return ep


def patch_logits(model, f, stype, mask, B, S, Z, G):
    """Per-patch study logits (B, S*G, F) and the validity mask (B, S*G, 1).

    f: (B*S, Z, D) backbone slice features (the series encoder's input).
    """
    if G < 1 or Z % G:
        raise ValueError(f"pat_grid={G} must divide n_slices={Z}")
    Zp = Z // G
    D = f.shape[-1]
    fp = f.reshape(B * S * G, Zp, D)            # slab g = slots [g*Zp,(g+1)*Zp)
    st = stype.reshape(B * S).repeat_interleave(G)
    _, tok = model.series_enc(fp, st, return_tokens=True)
    n_out = model.ev_b.shape[0]
    e = model.ev_head(tok).reshape(B * S * G, Zp, n_out).float()
    ep = ev_pool(model, e, Zp).reshape(B, S * G, n_out)
    lp = model.ev_b + ep
    valid = (mask.reshape(B, S, 1) > 0).repeat_interleave(G, dim=1)
    return lp, valid


def pat_aggregate(lp, valid):
    """Reference `weighted_sum` (self form): per-finding softmax over the valid
    patches of the patch logits, times the patch logits. (B, P, F) -> (B, F)."""
    a = torch.softmax(lp.masked_fill(~valid, float("-inf")), dim=1)
    return (a * lp.masked_fill(~valid, 0.0)).sum(1)
