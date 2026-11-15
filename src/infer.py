"""Inference: DICOM studies -> submission.csv.

Runs unchanged locally and inside the Kaggle notebook. This is a code
competition, so the notebook has no internet: backbone weights must come from
the checkpoint (never `pretrained=True`), and every model file ships as a
Kaggle Dataset attached to the notebook.

Preprocessing here must match `src/preprocess.py` exactly - same slice
ordering, same 1st-99th percentile window, same resize - or the model sees a
different distribution at test time than it trained on.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import FINDINGS, PLANES, STYPE_ID  # noqa: E402

MAX_SLICES = 48
SIZE = 384


def read_series_dir(d):
    """Directory of .dcm files -> (Z, 384, 384) uint8 volume."""
    import cv2
    import pydicom

    slices = []
    for f in os.listdir(d):
        if not f.endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True)
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception:
            continue
    if not slices:
        return None

    def key(ds):
        try:
            iop = [float(x) for x in ds.ImageOrientationPatient]
            ipp = [float(x) for x in ds.ImagePositionPatient]
            n = np.cross(np.array(iop[:3]), np.array(iop[3:]))
            return float(np.dot(np.array(ipp), n))
        except Exception:
            try:
                return float(ds.InstanceNumber)
            except Exception:
                return 0.0

    slices.sort(key=key)
    arrs = []
    for ds in slices:
        try:
            a = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        sl = float(getattr(ds, "RescaleSlope", 1) or 1)
        it = float(getattr(ds, "RescaleIntercept", 0) or 0)
        if a.ndim == 3:
            arrs.extend([f * sl + it for f in a])
        else:
            arrs.append(a * sl + it)
    if not arrs:
        return None

    if len(arrs) > MAX_SLICES:
        idx = np.linspace(0, len(arrs) - 1, MAX_SLICES).round().astype(int)
        arrs = [arrs[i] for i in idx]
    arrs = [cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for a in arrs]
    v = np.stack(arrs).astype(np.float32)
    lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.zeros(v.shape, np.uint8)
    return (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def prep_study(series_list, n_series, n_slices, size, side=""):
    """series_list: [(volume, plane, fluid_sensitive)] -> model tensors."""
    by_type = {}
    for vol, plane, fs in series_list:
        # Match src/data.py: mirror right knees on coronal/axial so "medial"
        # sits on a consistent side of the frame.
        if side == "R":
            if plane in ("Coronal", "Axial"):
                vol = np.ascontiguousarray(vol[:, :, ::-1])
            elif plane == "Sagittal":
                vol = np.ascontiguousarray(vol[::-1])
        by_type.setdefault(STYPE_ID[(plane, fs)], []).append(vol)
    order = sorted(by_type, key=lambda k: (k % 2, k // 2))
    picked, i = [], 0
    while len(picked) < n_series and any(by_type.values()):
        k = order[i % len(order)]
        if by_type[k]:
            picked.append((by_type[k].pop(), k))
        i += 1
        if i > 200:
            break

    vols = np.zeros((n_series, n_slices, size, size), np.float32)
    st = np.zeros((n_series,), np.int64)
    mask = np.zeros((n_series,), np.float32)
    for j, (vol, k) in enumerate(picked[:n_series]):
        z = vol.shape[0]
        idx = (np.linspace(0, z - 1, n_slices).round().astype(int) if z >= n_slices
               else np.clip(np.arange(n_slices) - (n_slices - z) // 2, 0, z - 1))
        v = vol[idx].astype(np.float32) / 255.0
        m = v.shape[1] - size
        if m > 0:
            v = v[:, m // 2:m // 2 + size, m // 2:m // 2 + size]
        vols[j] = v
        st[j] = k
        mask[j] = 1.0
    return (torch.from_numpy(vols)[None], torch.from_numpy(st)[None],
            torch.from_numpy(mask)[None])


def load_models(ckpt_paths, device):
    """Note: this CLI does not support the pulse-sequence embedding.

    Its prep_study never computes a sequence class, so a seq-aware checkpoint
    would silently run with every series tagged "unknown". Rather than produce
    quietly-wrong numbers it refuses them below. The Kaggle notebook is the
    shipped inference path and does support it.
    """
    from model import KneeNet
    models = []
    for p in ckpt_paths:
        ck = torch.load(p, map_location="cpu")
        a = ck["args"]
        sd = ck["model"]
        # Read the architecture off the weights. A checkpoint trained with the
        # report-text objective carries text_head.*, and one predating the
        # per-finding routing carries study_enc.pool.* - constructing the wrong
        # variant fails strict loading with a wall of key names.
        if "study_enc.pool.w.weight" in sd:
            raise SystemExit(
                f"{os.path.basename(p)} predates the per-finding routing "
                f"(study_enc.pool.*). src/model.py can no longer build that "
                f"architecture; score it with notebooks/kaggle_inference.py, "
                f"which keeps a legacy path.")
        text_dim = (sd["text_head.1.weight"].shape[0]
                    if "text_head.1.weight" in sd else 0)
        m = KneeNet(a["backbone"], pretrained=False, slice3d=a.get("slice3d", "none"), blurpool=a.get("blurpool", 0), qsa=a.get("qsa", "none"), qsa_gate_init=a.get("qsa_gate_init", 0.1),   # no internet on Kaggle
                    text_dim=text_dim)
        if "study_enc.seq_emb.weight" in sd:
            raise SystemExit(
                f"{os.path.basename(p)} was trained with --use-sequence, which "
                "src/infer.py cannot supply. Use the Kaggle notebook, or "
                "src/eval_checkpoints.py for gold scoring.")
        m.load_state_dict(sd, strict=True)
        models.append((m.to(device).eval(), a))
        print(f"loaded {os.path.basename(p)} "
              f"(fold-AUC {ck.get('fold_auc', float('nan')):.4f})", flush=True)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--series-csv", default=None,
                    help="defaults to <data-root>/test_series.csv")
    ap.add_argument("--test-csv", default=None)
    ap.add_argument("--series-dir", default=None)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--tta-flip", action="store_true")
    args = ap.parse_args()

    import csv
    root = args.data_root
    series_csv = args.series_csv or os.path.join(root, "test_series.csv")
    test_csv = args.test_csv or os.path.join(root, "test.csv")
    series_dir = args.series_dir or os.path.join(root, "test_series")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(args.ckpts, device)
    cfg = models[0][1]
    n_series, n_slices, size = cfg["n_series"], cfg["n_slices"], cfg["size"]

    index = {}
    with open(series_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            index.setdefault(r["StudyInstanceUID"], []).append(
                (r["SeriesInstanceUID"], r["Anatomical_Plane"],
                 int(r["Fluid_Sensitive"])))
    with open(test_csv, newline="", encoding="utf-8") as f:
        studies = [r["StudyInstanceUID"] for r in csv.DictReader(f)]
    print(f"{len(studies)} test studies", flush=True)

    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) \
        else torch.float16
    rows = []
    for n, study in enumerate(studies):
        series_list, side = [], ""
        for suid, plane, fs in index.get(study, []):
            d = os.path.join(series_dir, study, suid)
            if not os.path.isdir(d):
                continue
            v = read_series_dir(d)
            if v is not None:
                series_list.append((v, plane, fs))
            try:
                import pydicom, glob as _g
                _f = sorted(_g.glob(os.path.join(d, "*.dcm")))[:1]
                if _f:
                    _t = pydicom.dcmread(_f[0], stop_before_pixels=True,
                                         force=True).get((0x0020, 0x0060), None)
                    if _t is not None and str(_t.value).strip() and not side:
                        side = str(_t.value).strip().upper()[:1]
            except Exception:
                pass

        if not series_list:
            # No readable imaging: 0.5 across the board is the neutral choice
            # under AUC - any constant scores 0.5 and cannot hurt ranking.
            rows.append([study] + [0.5] * len(FINDINGS))
            print(f"  WARNING no series for {study}", flush=True)
            continue

        vol, st, mask = prep_study(series_list, n_series, n_slices, size, side)
        vol, st, mask = vol.to(device), st.to(device), mask.to(device)
        preds = []
        with torch.no_grad():
            for m, _ in models:
                with torch.autocast(device, dtype=amp, enabled=device == "cuda"):
                    # 4-tuple when the checkpoint carries a text head; the text
                    # output is training-only, so take the logits positionally.
                    lo = m(vol, st, mask)[0]
                preds.append(torch.sigmoid(lo.float()).cpu().numpy())
                # No flip TTA: mirroring a knee swaps medial and lateral.
        p = np.mean(preds, axis=0)[0]
        rows.append([study] + [float(x) for x in p])
        if n % 25 == 0:
            print(f"  {n}/{len(studies)}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["StudyInstanceUID"] + FINDINGS)
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
