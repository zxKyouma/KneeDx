"""Which findings does the image model rank BETTER than their own labels can?

The training targets come from report extraction. For each finding, the AUC of
the extracted grade against gold is a ceiling on what fitting those targets
perfectly can buy: a model that reproduces its targets exactly inherits their
ranking, and their ranking scores exactly that.

But the model does not start at its targets. It starts at ImageNet features,
which already rank some findings well - effusion is grossly visible, so is a
Baker's cyst. If a finding's epoch-0 gold AUC is ABOVE its label ceiling, then
training can only drag it down, and every additional epoch spent fitting those
targets costs us. That is the Synovitis story stated precisely: 0.82 at epoch 0
against a 0.741 label ceiling, decaying to 0.67.

This script pairs, per finding:

    ceiling   AUC(extracted grade -> gold)          - what the labels can buy
    peak      best per-epoch gold AUC in a run      - what the images can buy
    final     last-epoch gold AUC                   - what we actually ship

and flags findings where peak > ceiling (the model beat its labels) and
final < peak - margin (and we then threw it away). Those are exactly the
findings that want early stopping or a down-weighted loss, not more training.

Reads the per-finding lines the training loop already prints, so it needs no
GPU and no checkpoint. Usage:

    python src/label_ceiling.py logs/slurm-v2-58814847.out
    python src/label_ceiling.py logs/*.out --labels data/processed/report_labels_v2.jsonl
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# The training loop prints f.split()[0][:4], so "PF OA" -> "PF" and the
# Medial/Lateral pairs collide ("Medi", "Late" twice each). Position is the
# disambiguator: the loop always emits all twelve in FINDINGS order, so parse
# positionally and use the abbreviations only as a shape check.
PREFIX = [f.split()[0][:4] for f in FINDINGS]


def auc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    pos, neg = y == 1, y == 0
    n_p, n_n = pos.sum(), neg.sum()
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)


def label_ceilings(labels_path):
    """AUC of each extracted grade against gold - the per-finding label ceiling."""
    gold = json.load(open(PROC / "gold58.json"))
    grades = {}
    with open(labels_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                grades[r["uid"]] = r["grades"]
    covered = [g for g in gold if g["uid"] in grades]
    out = {}
    for f in FINDINGS:
        y = np.array([g["labels"][f] for g in covered])
        s = np.array([grades[g["uid"]].get(f, -1) for g in covered], dtype=float)
        out[f] = auc(y, s)
    return out, len(covered)


def parse_log(path):
    """Return list of per-epoch dicts {finding: gold_auc} in epoch order."""
    epochs = []
    pat = re.compile(r"gold per-finding:\s*(.+)")
    with open(path, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            # tokens look like "ACL:0.91 MCL:0.63 Medi:0.93 ..." - 12 of them
            toks = re.findall(r"([A-Za-z']+):([0-9.]+)", m.group(1))
            if len(toks) != 12:
                continue
            names = [t[0] for t in toks]
            if names != PREFIX:
                continue  # unexpected ordering: refuse rather than mislabel
            epochs.append({f: float(v) for f, (_, v) in zip(FINDINGS, toks)})
    return epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="training log(s) to read")
    ap.add_argument("--labels", default=str(PROC / "report_labels_v2.jsonl"))
    ap.add_argument("--margin", type=float, default=0.02,
                    help="decay below peak that counts as thrown away")
    args = ap.parse_args()

    paths = []
    for g in args.logs:
        paths.extend(sorted(glob.glob(g)))

    ceil, n_gold = label_ceilings(args.labels)
    print(f"label ceilings from {Path(args.labels).name} on {n_gold} gold studies\n")

    for p in paths:
        eps = parse_log(p)
        if not eps:
            continue
        print(f"=== {Path(p).name}  ({len(eps)} epochs logged) ===")
        print(f"{'finding':<18} {'ceiling':>8} {'ep0':>6} {'peak':>6} {'@ep':>4}"
              f" {'final':>6} {'peak-ceil':>10} {'lost':>7}")
        flagged = []
        for f in FINDINGS:
            traj = [e[f] for e in eps]
            peak = max(traj)
            k = int(np.argmax(traj))
            final = traj[-1]
            over = peak - ceil[f]
            lost = peak - final
            flag = ""
            if over > 0 and lost > args.margin:
                flag = "  <<<"
                flagged.append((f, over, lost, k, len(traj) - 1))
            print(f"{f:<18} {ceil[f]:>8.3f} {traj[0]:>6.3f} {peak:>6.3f}"
                  f" {k:>4} {final:>6.3f} {over:>+10.3f} {lost:>+7.3f}{flag}")

        if flagged:
            gain = sum(l for _, _, l, _, _ in flagged) / len(FINDINGS)
            print(f"\n  findings that beat their labels then decayed: "
                  f"{', '.join(f for f, _, _, _, _ in flagged)}")
            print(f"  recovering each finding's own peak epoch would add "
                  f"{gain:+.4f} to the mean of 12")
            print("  (ORACLE bound - the peak epoch is chosen on the same 58 "
                  "studies it is scored on)")
        else:
            print("\n  no finding both beat its label ceiling and decayed past "
                  f"the {args.margin} margin")
        print()


if __name__ == "__main__":
    main()
