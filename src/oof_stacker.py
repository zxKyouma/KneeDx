"""Can a second stage read the other eleven findings and improve the twelfth?

The twelve findings co-occur. Effusion and Synovitis lift each other, medial and
lateral OA travel together, a fracture implies contusion. The model predicts all
twelve from one shared trunk but through twelve INDEPENDENT heads, so nothing in
it can say "given how confident I am about Effusion, revise Synovitis".

This is the only untried idea in the family that has ever worked here. Every
mechanism story invented in this project has died on measurement; every gain -
EMA +0.008, rank-averaging two diverse models +0.005, folds + TTA +0.002, the
aux blend +0.006 - has been variance reduction or a better combiner. A stacker
is a better combiner. RSNA 2023's 2nd place built one and it was central to
their solution; their 3rd place tried a metric-optimising version and found it
WORSE "due to overfit". Both outcomes are live here.

An earlier version of this idea was already killed, and the difference matters.
Blending Effusion's predictions into Synovitis by hand was rejected because the
lift that motivated it was a gold-58 artifact: on 58 studies P(Syno|Effu) is
0.629 vs 0.217, but on the 4,406-study corpus it is only 0.316 vs 0.186. That
kill was correct AND it does not apply here, because this is fit on the ~4,344
out-of-fold studies where the weak-but-real corpus lift is what it will see.

Design, entirely shaped by not repeating the four false positives:

  * LINEAR, per finding, on rank-transformed inputs. Twelve coefficients per
    finding is the least freedom that can express "borrow from a correlate".
    Anything with capacity would fit the label noise, which is the failure mode
    3rd place reported.
  * NESTED cross-validation. The stacker is fit on K-1 folds and scored on the
    held-out one, rotated, so the reported number is out-of-sample for the
    stacker as well as for the base model. An in-sample stacker score is
    meaningless and would look excellent.
  * RIDGE with the penalty chosen inside the training split only.
  * The baseline it must beat is the base model's own column, scored on the
    same held-out rows. Not the base model's in-sample score.

If this does not clear its own baseline out-of-sample, it is dead, and that is
a useful thing to know cheaply - it costs no GPU and runs in seconds.

    python src/oof_stacker.py checkpoints/f5_convnext_tiny/fold*_oof.npz
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import FINDINGS
from label_ceiling import auc


def rank01(x):
    out = np.empty(x.shape, dtype=np.float64)
    n = x.shape[0]
    for j in range(x.shape[1]):
        col = x[:, j]
        order = np.argsort(col, kind="mergesort")
        r = np.empty(n, dtype=np.float64)
        r[order] = np.arange(n, dtype=np.float64)
        _, inv, cnt = np.unique(col, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        out[:, j] = (sums / cnt)[inv] / max(n - 1, 1)
    return out


def ridge(X, y, lam):
    """Closed-form ridge with an unpenalised intercept."""
    Xc = np.hstack([X, np.ones((len(X), 1))])
    A = Xc.T @ Xc
    P = np.eye(A.shape[0]) * lam
    P[-1, -1] = 0.0
    return np.linalg.solve(A + P, Xc.T @ y)


def apply_w(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", nargs="+")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lams", default="1,10,100,1000,10000")
    args = ap.parse_args()

    paths = []
    for g in args.oof:
        paths.extend(sorted(glob.glob(g)))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        raise SystemExit("no OOF files matched")

    Xs, Ys = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        # Rank WITHIN each source file. Folds select different epochs and are
        # differently calibrated, so pooling raw probabilities would let epoch
        # choice enter as signal.
        Xs.append(rank01(d["preds"]))
        Ys.append(d["targets"])
        print(f"  {os.path.basename(p):<22} {d['preds'].shape[0]} studies")
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    n = len(X)
    print(f"\npooled {n} out-of-fold studies, {X.shape[1]} input columns\n")

    lams = [float(v) for v in args.lams.split(",")]
    rng = np.random.RandomState(0)
    order = rng.permutation(n)
    base_auc, stack_auc, chosen = [], [], []

    for j, f in enumerate(FINDINGS):
        yb = (Y[:, j] >= 0.5).astype(int)
        pred = np.empty(n)
        lam_pick = []
        for k in range(args.folds):
            te = np.sort(order[k::args.folds])
            tr = np.sort(np.setdiff1d(order, te))
            # Choose lambda INSIDE the training split, on an inner split, so
            # the held-out rows never influence the model that scores them.
            inner = tr[: len(tr) // 2], tr[len(tr) // 2:]
            best = (-1.0, lams[0])
            for lam in lams:
                w = ridge(X[inner[0]], Y[inner[0], j], lam)
                v = auc((Y[inner[1], j] >= 0.5).astype(int),
                        apply_w(X[inner[1]], w))
                if v > best[0]:
                    best = (v, lam)
            lam_pick.append(best[1])
            w = ridge(X[tr], Y[tr, j], best[1])
            pred[te] = apply_w(X[te], w)
        b, s = auc(yb, X[:, j]), auc(yb, pred)
        base_auc.append(b)
        stack_auc.append(s)
        chosen.append(int(np.median(lam_pick)))

    print(f"{'finding':<18}{'base':>8}{'stacked':>9}{'delta':>8}{'lambda':>9}")
    for f, b, s, l in zip(FINDINGS, base_auc, stack_auc, chosen):
        print(f"{f:<18}{b:>8.4f}{s:>9.4f}{s - b:>+8.4f}{l:>9}")
    mb, ms = float(np.mean(base_auc)), float(np.mean(stack_auc))
    won = sum(s > b for b, s in zip(base_auc, stack_auc))
    print(f"{'-' * 52}\n{'MEAN':<18}{mb:>8.4f}{ms:>9.4f}{ms - mb:>+8.4f}")
    print(f"\nthe stacker beats the base column on {won}/12 findings.")
    print("Every number above is out-of-sample for the STACKER as well as for "
          "the base model: it is fit on K-1 splits and scored on the held-out "
          "one, with lambda chosen on an inner split. An in-sample stacker "
          "would look excellent and mean nothing.")

    if ms - mb <= 0:
        print("\n>>> DEAD. The findings do not carry usable information about "
              "each other beyond what the shared trunk already encodes.")
    else:
        # Where does the borrowing come from? Only worth reading if it won.
        print("\nlargest off-diagonal coefficients (fit on ALL rows - "
              "descriptive only, this is the in-sample model):")
        for j, f in enumerate(FINDINGS):
            if stack_auc[j] <= base_auc[j]:
                continue
            w = ridge(X, Y[:, j], chosen[j])[:-1]
            od = [(abs(w[i]), i) for i in range(len(FINDINGS)) if i != j]
            od.sort(reverse=True)
            top = ", ".join(f"{FINDINGS[i]} {w[i]:+.2f}" for _, i in od[:3])
            print(f"  {f:<18} self {w[j]:+.2f} | {top}")


if __name__ == "__main__":
    main()
