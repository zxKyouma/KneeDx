"""Regenerate out-of-fold predictions for a trained fold.

train.py now writes fold{N}_oof.npz beside the best checkpoint, but folds that
finished before that landed have only weights. Self-training needs OOF
predictions for every study - a model that saw a study in training gives an
in-sample prediction that reproduces the report label instead of correcting it,
which is the whole point of the exercise - so the missing folds have to be
recovered.

This reconstructs the fold split exactly as train.py does (same MinHash
grouping, same seed, same gold exclusion and gold-duplicate drop) and runs the
checkpoint over its own validation half. Any drift in that reconstruction would
silently mix in-sample studies into the "out-of-fold" set, so the split is
rebuilt from the same functions rather than reimplemented, and the study count
is printed for comparison against the training log's `fold N: train X val Y`.
"""

import argparse
import hashlib
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Fixed pooling rules over the auxiliary per-series head. `depth_pool_probe.py`
# measured these on the gold 58 and found the discarded aux head beats the
# learned study head on 7/12 findings - Synovitis +0.054, Fracture +0.037 - at
# zero extra inference cost, since the forward pass already computes them.
# 58 studies cannot arbitrate a per-finding choice among four pools (4^12
# degrees of freedom against a metric that cannot resolve 0.05), so they are
# dumped here alongside the main predictions and judged on the ~4,344 OOF
# studies instead. Nothing is selected at export time.
AUX_POOLS = ("mean_logit", "mean_prob", "top2_prob", "max_prob")


def evaluate_with_aux(model, loader, device, amp_dtype):
    """train.evaluate, plus fixed pools over the per-series aux logits.

    Kept here rather than folded into train.evaluate because that function runs
    every validation epoch of every job; this adds work no training run needs.
    """
    model.eval()
    P, Y, U = [], [], []
    pools = {k: [] for k in AUX_POOLS}
    with torch.no_grad():
        for b in loader:
            mask = b["mask"].to(device)
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=device != "cpu"):
                out = model(b["vol"].to(device, non_blocking=True),
                            b["stype"].to(device), mask,
                            b["seq"].to(device))
            P.append(torch.sigmoid(out[0].float()).cpu().numpy())
            aux = out[1].float()                       # (B, S, 12)
            mk = mask.unsqueeze(-1)                    # (B, S, 1)
            n = mk.sum(1).clamp(min=1)
            # A padded series must not enter any pool. -inf for the max/top-k,
            # excluded from the means; otherwise a study with fewer than
            # n_series series is dragged down on every finding, and studies
            # here have 3-14 series so that is most of them.
            pr = torch.sigmoid(aux)
            pr_m = pr.masked_fill(mk == 0, float("-inf"))
            pools["mean_logit"].append(((aux * mk).sum(1) / n).cpu().numpy())
            pools["mean_prob"].append(((pr * mk).sum(1) / n).cpu().numpy())
            # top-2 row-wise, not with a batch-wide k: a study with a single
            # valid series would otherwise average a real value with -inf, and
            # taking k from the batch minimum would make the result depend on
            # which studies happened to share a batch.
            # k cannot exceed the series axis: --pfrel serves S == 1 (one
            # axial FS series), and topk(2) on a length-1 axis raises
            # "selected index k out of range" AFTER a full forward pass
            # (pfrel T4 smoke 60155747). At S == 1 the has2 branch below
            # already selects t2[:, 0], so clamping k is exact, not a
            # degradation, for every roster that already worked.
            t2 = pr_m.topk(min(2, pr_m.shape[1]), dim=1).values   # (B, <=2, 12)
            has2 = (mask.sum(1) >= 2).view(-1, 1) & (t2.shape[1] >= 2)
            pools["top2_prob"].append(
                torch.where(has2, t2.mean(1), t2[:, 0]).cpu().numpy())
            pools["max_prob"].append(pr_m.max(1).values.cpu().numpy())
            if "y" in b:
                Y.append(b["y"].numpy())
            U += list(b["study"])
    P = np.concatenate(P)
    Y = np.concatenate(Y) if Y else None
    pools = {k: np.concatenate(v) for k, v in pools.items()}
    n_bad = int((~np.isfinite(P)).sum())
    if n_bad:
        print(f"   WARNING {n_bad}/{P.size} non-finite predictions "
              f"({n_bad / P.size:.2%}) - AUCs below are not meaningful",
              flush=True)
    for k, v in pools.items():
        nb = int((~np.isfinite(v)).sum())
        if nb:
            print(f"   WARNING aux pool {k}: {nb}/{v.size} non-finite",
                  flush=True)
    return P, Y, U, pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--syno-ablate", default="none",
                    choices=["none", "B", "C", "D", "all"],
                    help="--syno-specialist decomposition read; see "
                         "src/eval_checkpoints.py")
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--series-csv", default="data/raw/train_series.csv")
    ap.add_argument("--vol-root", default="data/volumes/train")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--all", action="store_true",
                    help="predict EVERY study, not just the held-out half. For "
                         "distillation targets: see the comment at the use site.")
    ap.add_argument("--smoke-limit", type=int, default=0,
                    help="SMOKE ONLY: truncate the held-out set to the first N "
                         "studies. Never use for a measurement - it refuses "
                         "unless --out contains 'smoke', so a truncated npz "
                         "can never be mistaken for a rig read.")
    ap.add_argument("--expect-val", type=int, default=0,
                    help="val-set size from the training log; the run "
                         "aborts if the rebuilt split disagrees")
    ap.add_argument("--dump-feats", default="",
                    help="oofranker (scratch/oofranker/PREREG.md Amendment 1): "
                         "also write finding-query embeddings (N,12,D) and "
                         "pooled per-series features (N,S,D) to this .npz, via "
                         "forward hooks on the study encoder. Default off; "
                         "registers nothing and changes no forward semantics.")
    ap.add_argument("--pick-best-twin", action="store_true",
                    help="SERVE-TIME data-path flag (scratch/twinserve): among "
                         "same-(plane,fluid) candidate series serve the "
                         "best-quality one. NOT read from ck['args'] - it is a "
                         "property of this run, not of the checkpoint.")
    ap.add_argument("--studies", default="",
                    help="Text file of StudyInstanceUIDs, one per line. Scores "
                         "exactly these studies instead of the rebuilt fold "
                         "val half. Required for all-data (n_folds=50) and "
                         "off-fold members that must land on one FIXED study "
                         "set; the split is still rebuilt and printed above.")
    ap.add_argument("--split-only", action="store_true",
                    help="rebuild and check the split, then exit before loading "
                         "any weights. CPU-only; verifies --expect-val agreement "
                         "without spending a GPU allocation.")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import (load_sequence, FINDINGS, KneeStudyDataset, fit_prob_table, fit_silent_table, load_zspacing, load_aspect,
                      gold_calib_split, load_geometry, load_gold, load_laterality,
                      load_report_labels, load_series_index, load_series_quality,
                      near_duplicate_groups)
    from model import KneeNet

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    fold, n_folds = a["fold"], a["n_folds"]
    print(f"[oof] {os.path.basename(args.ckpt)}  ep{ck.get('epoch')}  "
          f"fold {fold}/{n_folds}  calib={a.get('calib')}  "
          f"silent={a.get('silent_calib')}", flush=True)

    # --- rebuild the split with the checkpoint's OWN settings --------------
    gold = load_gold()
    calib_exclude = ()
    if a.get("calib") == "kfold":
        _f, held = gold_calib_split(gold, a["calib_fold"], a["calib_folds"])
        calib_exclude = held
    prob_table, _ = fit_prob_table(a["labels"], mode=a.get("calib", "fixed"),
                                   exclude=calib_exclude)
    silent = None
    if a.get("silent_calib") == "gold":
        silent, _ = fit_silent_table(a["labels"], pooled=prob_table.get(-1),
                                     k=a.get("silent_shrink", 10.0),
                                     exclude=calib_exclude)
    labels = load_report_labels(a["labels"], prob_table, silent)
    series_index = load_series_index(args.series_csv)
    lat = load_laterality(a.get("laterality_map",
                                "data/processed/laterality_map_v2.json"))
    geom = load_geometry() if a.get("use_geometry") else {}

    have = {s for s in labels if os.path.isdir(os.path.join(args.vol_root, s))}
    # train.py:1244 holds the external co-training pool OUT of `studies` so the
    # grouping, the group permutation and the val fold all behave as on the
    # competition pool alone; the pool is added back to `tr` after the split.
    # Omitting it here does not merely add extra rows - it changes the GROUP
    # PERMUTATION, so every fold assignment differs and nothing is out-of-fold.
    cotrain = set()
    if a.get("cotrain_pool"):
        with open(a["cotrain_pool"]) as f:
            for _v in json.load(f).values():
                cotrain |= set(_v)
        cotrain &= have
        print(f"[oof] cotrain pool {a['cotrain_pool']}: {len(cotrain)} external "
              f"studies held out of the split (as train.py does)", flush=True)
    studies = sorted(have - set(gold) - cotrain)
    csv.field_size_limit(10 ** 7)
    with open(args.train_csv, newline="", encoding="utf-8") as f:
        rep = {r["StudyInstanceUID"]: " ".join(r["Report"].split())
               for r in csv.DictReader(f)}
    rkey = lambda s: hashlib.md5(rep.get(s, s).encode()).hexdigest()
    if not a.get("keep_gold_dupes"):
        gt = {rkey(s) for s in gold if s in rep}
        studies = [s for s in studies if rkey(s) not in gt]

    # Rebuild the grouping the checkpoint ACTUALLY used. Defaulting to 0.8 for
    # a checkpoint that predates MinHash would produce a different split, and
    # studies that were in its training half would silently land in the "out of
    # fold" set - the exact contamination this file exists to avoid.
    if "group_threshold" in a:
        gkey = near_duplicate_groups({s: rep.get(s, s) for s in studies},
                                     threshold=a["group_threshold"])
        how = f"minhash@{a['group_threshold']}"
    else:
        gkey = {s: rkey(s) for s in studies}
        how = "exact-hash (checkpoint predates MinHash grouping)"
    groups = sorted({gkey[s] for s in studies})
    gperm = np.random.RandomState(a.get("split_seed", 42)).permutation(len(groups))
    val_groups = {groups[i] for i in gperm[fold::n_folds].tolist()}
    va = [s for s in studies if gkey[s] in val_groups]
    print(f"[oof] rebuilt split ({how}): val {len(va)} of {len(studies)}",
          flush=True)
    if args.expect_val and args.expect_val != len(va):
        raise SystemExit(
            f"[oof] ABORT: rebuilt val set is {len(va)} studies but the training "
            f"log says {args.expect_val}. The split does not reproduce, so these "
            f"would not be out-of-fold predictions. Do not use them.")
    if args.split_only:
        print(f"[oof] SPLIT_OK {args.ckpt} val={len(va)}", flush=True)
        return
    if args.studies:
        want = [ln.strip() for ln in open(args.studies) if ln.strip()]
        miss = [u for u in want if u not in have]
        if miss:
            raise SystemExit(f"[oof] --studies: {len(miss)} uid(s) absent from "
                             f"{args.vol_root}/labels, e.g. {miss[:3]}")
        va = want
        print(f"[oof] --studies {args.studies}: FIXED set of {len(va)} studies "
              f"(the rebuilt fold val half is NOT used; in-sample rows are "
              f"possible and are the caller's responsibility)", flush=True)
    if args.all:
        # Distillation, not stacking. A student validating on fold k must take
        # every target from the fold-k TEACHER, which never saw a fold-k study
        # - so we need that teacher's predictions over ALL studies, including
        # the ones it trained on. Those in-sample predictions are what
        # classical KD distils from and are NOT a leak: the leak would be
        # mixing five teachers, which is what the OOF file does.
        #
        # The split is still rebuilt and still checked above, because
        # --expect-val is the only evidence that this checkpoint's fold is what
        # its args claim - and build_distill_targets.py refuses to run if the
        # teacher's fold does not match the student's val fold.
        va = list(studies)
        print(f"[oof] --all: predicting every study ({len(va)}), for "
              f"distillation targets rather than stacking", flush=True)
    if not args.expect_val:
        print("[oof] WARNING no --expect-val given. Pass the 'val N' from the "
              "training log so a split mismatch fails loudly instead of "
              "producing contaminated predictions.", flush=True)

    # --- run the checkpoint over its own held-out half --------------------
    # Model construction is src/ckpt_build.py, the SINGLE path shared with
    # src/eval_checkpoints.py (which had its own drifted copy - see
    # scratch/measaudit/GOLDSKIP_AUDIT_20260914.md). The helper resolves the
    # same flags this block resolved, from ck["args"] with the train.py default
    # as fallback, plus n_out and moaks_blocks which only the other copy had;
    # for every checkpoint this exporter could already build, the constructed
    # model is identical.
    if args.smoke_limit:
        if "smoke" not in os.path.basename(args.out):
            raise SystemExit("--smoke-limit requires 'smoke' in --out")
        print(f"[SMOKE] truncating held-out set {len(va)} -> "
              f"{args.smoke_limit}; this npz is NOT a rig read", flush=True)
        va = va[: args.smoke_limit]
    from ckpt_build import build_from_checkpoint
    sd = ck["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _a, _meta = build_from_checkpoint(
        args.ckpt, device=device, ck=ck,
        syno_ablate=getattr(args, "syno_ablate", "none"))

    # KNEE_NSLICES: inference-only window-count override (scratch/densinf).
    # Unset/0 -> the checkpoint's own n_slices, byte-identical behaviour.
    _ns = int(os.environ.get("KNEE_NSLICES", "0") or 0) or a["n_slices"]
    if _ns != a["n_slices"]:
        print(f"[densinf] n_slices override {a['n_slices']} -> {_ns}")
    # --tfcrop-root (scratch/tfcrop_20261011): data path from ck["args"].
    KneeStudyDataset.TFCROP_ROOT = str(a.get("tfcrop_root", "") or "")
    if KneeStudyDataset.TFCROP_ROOT and os.environ.get("KNEE_TFCROP_ABLATE") == "1":
        # crop-ablation read: crop slots present but masked (no npz found -> mask 0)
        KneeStudyDataset.TFCROP_ROOT = "/nonexistent_tfcrop_ablate"
        print("[oof] tfcrop ABLATE: crop slots masked", flush=True)
    if KneeStudyDataset.TFCROP_ROOT:
        print(f"[oof] tfcrop_root={KneeStudyDataset.TFCROP_ROOT}", flush=True)
    ds = KneeStudyDataset(va, series_index, labels, args.vol_root,
                          a["n_series"], _ns, a["size"], False,
                          laterality=lat, geometry=geom,
                          sequence=load_sequence() if a.get('use_sequence') else {},
                          pfrel_series=(str(a.get("pfrel", "off")) != "off"), bone_series=(str(a.get("bone", "off")) != "off"), bone_fluid=(0 if str(a.get("bone_series_fluid", "fs")) == "nonfs" else 1),
                          native_ctx=bool(a.get("native_ctx", False)), slice_ctx=a.get("slice_ctx", 3),
                          ctx_mm=a.get("ctx_mm", 0.0), zspacing=(load_zspacing() if a.get("ctx_mm", 0) else None),
                          aspect=(load_aspect() if a.get("aspect_fix") else None),
                          full_fov=bool(a.get("full_fov", False)), xfuse_pairs=(__import__("data").load_xfuse_pairs(a.get("xfuse_pairs") or "scratch/xfuse/xfuse_pairs.json") if a.get("xfuse") else None),
                          pick_best_twin=args.pick_best_twin,
                          series_quality=(load_series_quality()
                                          if args.pick_best_twin else None))
    print(f"[oof] pick_best_twin={'ON' if args.pick_best_twin else 'OFF'}", flush=True)
    dl = DataLoader(ds, batch_size=a["bs"], shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    amp = torch.bfloat16 if (device == "cuda"
                             and torch.cuda.get_device_capability(0)[0] >= 8) \
        else torch.float16
    feats = {}
    hook = None
    if args.dump_feats:
        # Hooks only. study_enc.forward(series_emb, stype, mask, ...) -> (study_emb, attn)
        # input[0]  = pooled per-series features (B, S, D)  [attn-pooled slices]
        # output[0] = study-transformer query embeddings (B, n_out(+aux), D);
        #             the scored twelve are the leading rows (see model.py:1923).
        feats = {"q": [], "sf": [], "sm": []}

        def _cap(mod, inp, out):
            se = inp[0].detach().float()
            mk = inp[2].detach()
            qe = (out[0] if isinstance(out, (tuple, list)) else out)
            qe = qe.detach().float()[:, :12]
            feats["q"].append(qe.cpu().numpy().astype(np.float16))
            feats["sf"].append(se.cpu().numpy().astype(np.float16))
            feats["sm"].append(mk.cpu().numpy().astype(np.uint8))

        hook = model.study_enc.register_forward_hook(_cap)
        print(f"[feat] hook on study_enc; dumping to {args.dump_feats}", flush=True)

    P, Y, U, pools = evaluate_with_aux(model, dl, device, amp)
    if hook is not None:
        hook.remove()
        F = {k: np.concatenate(v) for k, v in feats.items()}
        assert len(F["q"]) == len(U), (F["q"].shape, len(U))
        np.savez_compressed(args.dump_feats, uids=np.array(U),
                            queries=F["q"], series_feat=F["sf"],
                            series_mask=F["sm"],
                            probs=P.astype(np.float32),
                            targets=(Y.astype(np.float32) if Y is not None
                                     else np.zeros((len(U), 12), np.float32)),
                            fold=fold, ckpt=os.path.abspath(args.ckpt))
        print(f"[feat] wrote {args.dump_feats} queries={F['q'].shape} "
              f"series_feat={F['sf'].shape}", flush=True)

    out = args.out or os.path.join(os.path.dirname(args.ckpt),
                                   f"fold{fold}_oof.npz")
    np.savez_compressed(out, uids=np.array(U), preds=P.astype(np.float32),
                        targets=Y.astype(np.float32), epoch=ck.get("epoch"),
                        fold=fold,
                        **{f"aux_{k}": v.astype(np.float32)
                           for k, v in pools.items()})
    from train import mean_auc
    m, _ = mean_auc(P, Y)
    print(f"[oof] wrote {out}  {len(U)} studies  fold-AUC@.5 {m:.4f} "
          f"(training log said {ck.get('fold_auc', float('nan')):.4f} on the "
          f"selection metric, a different quantity)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
