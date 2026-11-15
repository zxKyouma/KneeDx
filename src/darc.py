"""DARC-local arm (scratch/shortcutarms/PREREG.md §3) - HELD, launch only on user go.

Local counterfactual stream of DARC (Liao et al., CVPR 2026), adapted to the EV head.
The per-slice evidence of finding k is e_k = W_k . h + b_k (h = the normalised slice
token, W = ev_head's Linear rows). The counterfactual input for target k keeps only
the part of h that the COMPANION readouts can see, h_bar_k = P_C(k) h (orthogonal
projection onto span{W_j : j != k}), so the total direct effect is

    TDE_k = W_k . h - W_k . h_bar_k = ((I - P_C(k)) W_k) . h

i.e. finding k is read along its own direction with every companion direction
projected out. Trained AND served on the TDE evidence (the paper's local stream trains
on the direct-effect logit). Companion rows enter P under stop-gradient; the Synovitis
row is kept verbatim (placebo). Zero parameters; the transformed rows are a pure
function of ev_head.weight, so an export-time fold (U -> ev_head.weight) would serve in
the existing kernel unchanged.
"""
import torch


def darc_rows(W, keep, ridge=1e-6):
    """W (F, D) -> U (F, D): row k = (I - P_{rows != k, detached}) W_k; keep rows verbatim."""
    with torch.autocast(device_type=W.device.type, enabled=False):
        F_, _ = W.shape
        W64 = W.double()
        Wd = W64.detach()
        idx = torch.tensor([[j for j in range(F_) if j != k] for k in range(F_)],
                           device=W.device)
        C = Wd[idx]                                           # (F, F-1, D)
        G = C @ C.transpose(1, 2)                             # (F, F-1, F-1)
        lam = ridge * G.diagonal(dim1=1, dim2=2).mean(-1)
        G = G + lam[:, None, None] * torch.eye(F_ - 1, dtype=G.dtype, device=G.device)
        coef = torch.linalg.solve(G, C @ W64.unsqueeze(-1))   # (F, F-1, 1)
        U = W64 - (C.transpose(1, 2) @ coef).squeeze(-1)
        U = torch.where(keep[:, None], W64, U)
        return U.to(W.dtype)
