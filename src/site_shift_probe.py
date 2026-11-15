"""Is our score hostage to the test set's institutional mix?

The reason to ask. RSNA built the 2021 brain-tumour private test set to
deliberately over-represent institutions ABSENT from training, and disclosed it
only after the competition closed - the chair framed it as a methodological
contribution, not an accident. Same organiser; this competition spans 16
institutions and ~5,000 exams. Our grouped CV groups by MinHash near-duplicate
REPORT TEXT, which is an axis chosen to stop the same study leaking across a
fold boundary. It was never chosen to separate sites, and nothing has ever
checked whether it does.

That matters more than usual right now: we lead the public board by 0.002, and
every selection we have made - checkpoints, ensemble members, the +0.026
gold-to-LB offset we keep quoting - assumes the public split predicts the
private one.

Three questions, all answerable from data already on disk, no GPU:

1. DOES PERFORMANCE VARY BY SITE? Per-group AUC over the 4,344 out-of-fold
   studies, grouped by scanner manufacturer, field strength, and a
   series-description signature that acts as a site fingerprint (naming
   conventions are institution-specific and survive anonymisation).

2. HOW SENSITIVE IS OUR SCORE TO THE MIX? If the per-site AUCs differ, then a
   test set weighted differently from the training corpus scores differently
   for reasons that have nothing to do with the model. This reports the spread
   directly: best-case and worst-case mean AUC under a re-weighting of sites.

3. DOES OUR FOLD GROUPING ALREADY SEPARATE SITES? If report-text MinHash groups
   correlate with site, we are accidentally partly site-grouped and the risk is
   smaller. If they are independent, every fold model saw every site, and our
   CV cannot see site generalisation at all.

What this CANNOT do: measure leave-one-site-out generalisation. Every fold model
saw every site, so the OOF predictions are in-distribution for site by
construction. Answering that needs a training run with a site held out. This
probe decides whether that run is worth a slot - it sizes the exposure, it does
not measure the effect.

    python src/site_shift_probe.py checkpoints/f5_convnext_tiny/fold*_oof.npz
"""

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import FINDINGS
from label_ceiling import auc


def desc_signature(desc):
    """A site fingerprint from SeriesDescription conventions.

    Institutions name series in house styles that survive anonymisation -
    separator characters, casing, whether the plane comes first, whether the
    protocol is abbreviated. Strip the content words and what is left is the
    convention, not the sequence.
    """
    d = (desc or "").strip()
    if not d:
        return "empty"
    sep = "_" if "_" in d else ("." if "." in d else (" " if " " in d else "none"))
    case = ("upper" if d.isupper() else "lower" if d.islower() else "mixed")
    return f"{sep}|{case}|{'digits' if re.search(r'[0-9]', d) else 'nodigits'}"


def load_site_meta(path="data/processed/series_laterality.json"):
    """{study: {manufacturer, field, desc_sig}} - the site proxies we have."""
    rows = json.load(open(path))
    per = collections.defaultdict(list)
    for r in rows:
        per[r["study"]].append(r)
    out = {}
    for s, rs in per.items():
        man = collections.Counter(
            (r.get("manufacturer") or "unknown").strip() or "unknown" for r in rs)
        fld = collections.Counter(str(r.get("field") or "?") for r in rs)
        sig = collections.Counter(desc_signature(r.get("series_desc")) for r in rs)
        out[s] = {
            "manufacturer": man.most_common(1)[0][0],
            "field": fld.most_common(1)[0][0],
            "desc_sig": sig.most_common(1)[0][0],
        }
    return out


def mean_auc(P, Y, rows):
    if rows.sum() < 30:
        return float("nan")
    vals = []
    for j in range(len(FINDINGS)):
        y = (Y[rows, j] >= 0.5).astype(int)
        if y.sum() < 3 or (1 - y).sum() < 3:
            continue
        vals.append(auc(y, P[rows, j]))
    return float(np.nanmean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", nargs="+")
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--perm", type=int, default=200,
                    help="permutations for the same-sizes null")
    args = ap.parse_args()

    paths = []
    for g in args.oof:
        paths.extend(sorted(glob.glob(g)))
    paths = [p for p in paths if os.path.isfile(p)]

    U, P, Y, F = [], [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        # Rank within fold: folds pick different epochs and are differently
        # calibrated, so raw probabilities are not comparable across them and a
        # per-site AUC pooled over folds would partly measure epoch choice.
        pr = d["preds"]
        rk = np.empty(pr.shape)
        for j in range(pr.shape[1]):
            o = np.argsort(pr[:, j], kind="mergesort")
            r = np.empty(len(o)); r[o] = np.arange(len(o))
            rk[:, j] = r / max(len(o) - 1, 1)
        U += [str(u) for u in d["uids"]]
        P.append(rk); Y.append(d["targets"])
        F += [int(d["fold"])] * len(d["uids"])
    P, Y, F = np.concatenate(P), np.concatenate(Y), np.array(F)
    meta = load_site_meta()
    print(f"{len(U)} out-of-fold studies over {len(paths)} folds\n")

    overall = mean_auc(P, Y, np.ones(len(U), bool))

    # ---- 1 & 2: per-site AUC and the mix sensitivity ------------------------
    for key in ("manufacturer", "field", "desc_sig"):
        groups = collections.defaultdict(list)
        for i, u in enumerate(U):
            groups[meta.get(u, {}).get(key, "unknown")].append(i)
        big = {k: v for k, v in groups.items() if len(v) >= args.min_n}
        if len(big) < 2:
            continue
        print(f"=== by {key} (groups with n >= {args.min_n}) ===")
        print(f"{'group':<28}{'n':>7}{'AUC':>9}{'vs all':>9}")
        vals = {}
        for k, idx in sorted(big.items(), key=lambda kv: -len(kv[1])):
            rows = np.zeros(len(U), bool); rows[idx] = True
            a = mean_auc(P, Y, rows)
            vals[k] = (a, len(idx))
            print(f"{str(k)[:27]:<28}{len(idx):>7}{a:>9.4f}{a - overall:>+9.4f}")
        aa = np.array([v[0] for v in vals.values()])
        aa = aa[np.isfinite(aa)]
        if len(aa) >= 2:
            obs = aa.max() - aa.min()
            # THE CONTROL. Groups of a few hundred studies have wide AUC error
            # bars, so ANY partition of this size produces a spread. Permute the
            # group labels, keeping the group SIZES exactly, and see how often
            # chance alone beats the observed spread. Without this the number
            # below is just "small groups are noisy" wearing a site-shift
            # costume - which is the shape of four earlier false positives here.
            sizes = [len(v) for v in big.values()]
            rng = np.random.RandomState(0)
            null = []
            for _ in range(args.perm):
                perm = rng.permutation(len(U))
                lo = 0; sub = []
                for n_ in sizes:
                    rows = np.zeros(len(U), bool); rows[perm[lo:lo + n_]] = True
                    lo += n_
                    sub.append(mean_auc(P, Y, rows))
                sub = np.array(sub); sub = sub[np.isfinite(sub)]
                if len(sub) >= 2:
                    null.append(sub.max() - sub.min())
            null = np.array(null)
            pval = float((null >= obs).mean()) if len(null) else float("nan")
            print(f"{'':28}{'spread':>7}{obs:>9.4f}")
            print(f"  best group {aa.max():.4f}  worst {aa.min():.4f}  "
                  f"pooled {overall:.4f}")
            print(f"  RANDOM partitions of the same sizes span "
                  f"{null.mean():.4f} on average (95th pct {np.percentile(null, 95):.4f});"
                  f"  p = {pval:.3f}")
            if pval > 0.05:
                print("  -> NOT distinguishable from chance. This grouping does "
                      "not show site-specific weakness.\n")
            else:
                print("  -> larger than chance. The mix genuinely moves the "
                      "score.\n")

    # ---- 3: does the fold grouping already separate sites? ------------------
    print("=== does MinHash report-text grouping separate sites? ===")
    for key in ("manufacturer", "desc_sig"):
        # If folds were site-separated, each site would concentrate in one fold.
        tab = collections.defaultdict(lambda: collections.Counter())
        for i, u in enumerate(U):
            tab[meta.get(u, {}).get(key, "unknown")][F[i]] += 1
        conc, tot = [], 0
        for k, c in tab.items():
            n = sum(c.values())
            if n < args.min_n:
                continue
            conc.append(max(c.values()) / n)
            tot += 1
        if conc:
            exp = 1.0 / len(set(F.tolist()))
            print(f"  {key:<14} mean largest-fold share {np.mean(conc):.3f} "
                  f"(random would be {exp:.3f}) over {tot} groups")
    print("\n  A share near the random baseline means every fold model saw every "
          "site, so our CV cannot measure site generalisation at all - the "
          "grouping is orthogonal to the axis the private split may use.")
    print("\nThis probe SIZES the exposure. It cannot measure leave-one-site-out "
          "generalisation, because every fold model saw every site by "
          "construction; that needs a training run with a site held out.")


if __name__ == "__main__":
    main()
