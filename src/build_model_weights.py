"""Emit model_weights.json: which checkpoints carry which findings.

The temptation here is to take, per finding, the argmax over checkpoints of
gold-58 AUC. Do not. Synovitis has ~27 positives in 58 studies, a bootstrap CI
width of 0.16-0.20, and an argmax over N checkpoints is a max of N draws from
that noise - it will look like a gain on the set it was chosen on and vanish on
the leaderboard. That is the same mistake the extraction prompt tuning made, and
`eval_extraction_cv.py` exists because of it.

So this script does not search. It applies ONE structural rule that five
independent training runs agree on and that has a measured mechanism behind it:

    A finding whose LABEL CEILING - AUC(extracted grade -> gold) - sits below
    what the model reaches early is dragged DOWN by further training, because
    BCE converges to its targets. Those findings want an early checkpoint.
    Everything else wants the selected-best checkpoint.

Synovitis is the clear case: ceiling 0.723, epoch-0 gold 0.79-0.84, epoch-11
0.67-0.69, peaking at epoch 0 in 5/5 arms. `--auto` widens the rule to any
finding whose measured early-vs-late gap exceeds `--min-gap`, but the default is
Synovitis alone, because that is the only one where the mechanism was
established independently of the numbers being used to select.

Weights are per (finding, checkpoint) and default to 1.0, so the shipped file
only ever names what it deliberately changes.

    python src/build_model_weights.py data/processed/ckpt_gold.npz \
        --late "fold*_ema.pt" --out data/processed/model_weights.json
"""

import argparse
import fnmatch
import json
import re

import numpy as np

# Established independently of this script's inputs: peaks at epoch 0 in arms
# A, B, D and folds 0-3 of the 5-fold, with the ceiling argument explaining why.
EARLY_FINDINGS = ["Synovitis"]


def epoch_of(name):
    """Epoch encoded by scripts/snapshot_epochs.sh as ..._ep{N}; None if absent.

    Names here are the flattened upload names (<run>_<file>), which is what the
    notebook sees after scripts/upload_weights.sh stages the dataset - not the
    local basename. A mismatch is invisible: the notebook treats an unknown
    name as weight 1.0 and prints only a passing note, so the whole file
    degrades to the plain rank average while appearing to have been applied.
    """
    m = re.search(r"_ep(\d+)$", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="output of src/eval_checkpoints.py")
    ap.add_argument("--out", default="data/processed/model_weights.json")
    ap.add_argument("--early-max-epoch", type=int, default=2,
                    help="checkpoints at or below this epoch count as 'early'")
    ap.add_argument("--auto", action="store_true",
                    help="also route any finding whose early-vs-late gap exceeds "
                         "--min-gap. Off by default: that is selection on 58 "
                         "studies and it will overfit.")
    ap.add_argument("--min-gap", type=float, default=0.05)
    ap.add_argument("--ship", default="*",
                    help="glob of checkpoint names actually shipped in the "
                         "weights dataset; others are ignored")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    findings = [str(x) for x in d["findings"]]
    A = d["auc"]                                   # (n_ckpt, 12)

    keep = [i for i, n in enumerate(names) if fnmatch.fnmatch(n, args.ship)]
    if not keep:
        raise SystemExit(f"--ship {args.ship!r} matched none of {names}")

    early = [i for i in keep
             if (epoch_of(names[i]) is not None
                 and epoch_of(names[i]) <= args.early_max_epoch)]
    late = [i for i in keep if i not in early]
    if not early:
        raise SystemExit("no early checkpoints among the shipped set; nothing to route")
    if not late:
        raise SystemExit("no late checkpoints among the shipped set; nothing to route")

    print(f"early ({len(early)}): {[names[i] for i in early]}")
    print(f"late  ({len(late)}): {[names[i] for i in late]}")

    print(f"\n{'finding':<18} {'early':>7} {'late':>7} {'gap':>7}   routed to")
    spec, routed = {}, []
    for k, f in enumerate(findings):
        e = float(np.nanmean(A[early, k]))
        l = float(np.nanmean(A[late, k]))
        rule = f in EARLY_FINDINGS or (args.auto and e - l > args.min_gap)
        if rule:
            # Zero the late checkpoints for this finding, keep the early ones.
            spec[f] = {names[i]: 0.0 for i in late}
            spec[f].update({names[i]: 1.0 for i in early})
            routed.append((f, e - l))
            where = "EARLY"
        else:
            # Early snapshots must NOT default to 1.0 here: uniform snapshot
            # ensembling measured NEGATIVE (0.9118 -> 0.8966), so an early
            # checkpoint that ships for Synovitis has to be explicitly zeroed
            # on every finding it is not routed to.
            spec[f] = {names[i]: 0.0 for i in early}
            where = "late (default)"
        print(f"{f:<18} {e:>7.3f} {l:>7.3f} {e-l:>+7.3f}   {where}")

    with open(args.out, "w") as fh:
        json.dump(spec, fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out} covering {len(spec)} finding(s)")

    if routed:
        gain = sum(g for _, g in routed) / len(findings)
        print(f"implied change on the mean of 12: {gain:+.4f}, measured on the "
              f"same 58 studies - directional only")
    print("\nEverything not named above keeps weight 1.0, i.e. the plain rank "
          "average. Verify by re-running the notebook with the file absent: the "
          "submission must be bit-identical to the unweighted one.")


if __name__ == "__main__":
    main()
