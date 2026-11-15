"""Turn raw DICOM series into compact uint8 volumes for training.

Reads members **directly out of `archive.zip`** rather than unzipping first.
The bundle is 265 GB across ~730k tiny DICOMs; unpacking that onto a Lustre
filesystem would cost a second 265 GB, hours of metadata-bound I/O, and make
every later epoch slower. `zipfile` supports random access via the central
directory, and each worker holds its own handle, so we skip all of it.

Output: one `.npy` per series at `{out}/{StudyUID}/{SeriesUID}.npy`, shape
(Z, 384, 384), uint8, plus a parquet of per-series geometry.

Design notes:
- In-plane 384 preserves detail for meniscal tears and cartilage, the findings
  that live or die on resolution. MEASURED 2026-08-12 and it is not a close
  call: median ACQUIRED matrix is 320x268 (the 512 recon is zero-fill), so
  384 already oversamples the true resolution by 1.16-1.44x and only 14.3% of
  series are coarser at 384 than their own fine axis. See --size.
- The resize is UNCONDITIONAL to (size, size). 1,125 series (758 studies, 17%)
  have Rows != Cols with isotropic PixelSpacing, so their anatomy is
  geometrically squashed by up to 2x - measured in the stored volumes as a
  tissue-bbox aspect of 1.964 on distorted axial against 1.012 on clean. It is
  vendor-correlated (Siemens/Toshiba/Canon; Philips and GE zero) and costs
  nothing measurable on OOF (p=0.63), because train and inference distort
  identically and the model has learned each vendor's warp. Left as-is
  deliberately: fixing it is site-robustness insurance, not a pooled gain, and
  it would invalidate every existing checkpoint's input contract.
- Slices stay at native count (capped at 48, uniformly subsampled beyond that)
  rather than being resampled to a fixed Z, since sagittal and axial series
  differ a lot and the model pools over Z anyway.
- Intensity is normalised per series on the 1st-99th percentile. MRI has no
  absolute scale like CT Hounsfield units and these come from 16 institutions,
  so a per-series robust window is the only sane common ground.
- Slice order comes from projecting ImagePositionPatient onto the slice normal.
  InstanceNumber is a fallback only; it is unreliable across vendors.
"""

import argparse
import collections
import io
import os
import sys
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

MAX_SLICES = 48
SIZE = 384
# Intensity window, as percentiles of the series. These were hardcoded until
# 2026-08-12, which meant a volume root carried no record of how it was built
# and a checkpoint carried no record of which root it trained on. A re-windowed
# model was therefore INDISTINGUISHABLE from an old one by inspection, and the
# inference notebook - which re-implements this transform - would have fed it
# the old window silently. Nothing would have raised.
LO_PCT = 1.0
HI_PCT = 99.0
MANIFEST = "_preprocess.json"

_ZIP = None  # per-worker handle, opened lazily


def _zip(path):
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(path)
    return _ZIP


def _read_dicoms(blobs):
    import pydicom

    slices = []
    for b in blobs:
        try:
            d = pydicom.dcmread(io.BytesIO(b), force=True)
            if hasattr(d, "PixelData"):
                slices.append(d)
        except Exception:
            continue
    if not slices:
        return None, {}

    def key(d):
        try:
            iop = [float(x) for x in d.ImageOrientationPatient]
            ipp = [float(x) for x in d.ImagePositionPatient]
            n = np.cross(np.array(iop[:3]), np.array(iop[3:]))
            return float(np.dot(np.array(ipp), n))
        except Exception:
            try:
                return float(d.InstanceNumber)
            except Exception:
                return 0.0

    slices.sort(key=key)

    arrs = []
    for d in slices:
        try:
            a = d.pixel_array.astype(np.float32)
        except Exception:
            continue
        slope = float(getattr(d, "RescaleSlope", 1) or 1)
        inter = float(getattr(d, "RescaleIntercept", 0) or 0)
        if a.ndim == 3:                      # multi-frame -> frames as slices
            for f in a:
                arrs.append(f * slope + inter)
        else:
            arrs.append(a * slope + inter)
    if not arrs:
        return None, {}

    ref = slices[0]
    meta = {
        "rows": int(getattr(ref, "Rows", 0)),
        "cols": int(getattr(ref, "Columns", 0)),
        "n_slices": len(arrs),
        "thickness": float(getattr(ref, "SliceThickness", 0) or 0),
        "pixel_spacing": float((getattr(ref, "PixelSpacing", [0, 0]) or [0, 0])[0]),
        "field": float(getattr(ref, "MagneticFieldStrength", 0) or 0),
        "series_desc": str(getattr(ref, "SeriesDescription", ""))[:80],
        "manufacturer": str(getattr(ref, "Manufacturer", ""))[:40],
        "tr": float(getattr(ref, "RepetitionTime", 0) or 0),
        "te": float(getattr(ref, "EchoTime", 0) or 0),
    }
    return arrs, meta


def _to_volume(arrs):
    import cv2

    if len(arrs) > MAX_SLICES:
        idx = np.linspace(0, len(arrs) - 1, MAX_SLICES).round().astype(int)
        arrs = [arrs[i] for i in idx]
    arrs = [cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for a in arrs]

    v = np.stack(arrs).astype(np.float32)
    # ONE partition instead of two. np.percentile partitions the whole volume
    # per call, and this is the single most expensive step in preprocessing -
    # measured 86 ms of a 158 ms series at 30 frames, against 25 ms for the
    # DICOM read and decode combined. Passing both quantiles together halves
    # the partitions.
    #
    # The .astype is load-bearing, not cosmetic: with a scalar q numpy returns
    # the input dtype (float32 here), with a sequence it returns float64.
    # Without the cast the window differs in the last ulp and every pixel
    # shifts. Verified bit-identical to the two-call form over 300 random
    # float32 volumes of varying depth and scale.
    lo, hi = np.percentile(v, (LO_PCT, HI_PCT)).astype(np.float32)
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.zeros(v.shape, np.uint8)
    return (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def process_series(job):
    zip_path, study, series, members, out_dir = job
    out_path = os.path.join(out_dir, study, series + ".npy")
    if os.path.exists(out_path):
        return {"study": study, "series": series, "status": "cached"}
    try:
        zf = _zip(zip_path)
        blobs = []
        for m in members:
            try:
                blobs.append(zf.read(m))
            except Exception:
                continue
        arrs, meta = _read_dicoms(blobs)
        if not arrs:
            return {"study": study, "series": series, "status": "empty"}
        vol = _to_volume(arrs)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Write through a file handle: np.save() appends ".npy" to a *path*
        # whose name does not already end in it, which silently renames the
        # temp file out from under the os.replace() below.
        tmp = out_path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, vol)
        os.replace(tmp, out_path)
        meta.update({"study": study, "series": series,
                     "status": "ok", "z": int(vol.shape[0])})
        return meta
    except Exception:
        return {"study": study, "series": series, "status": "error",
                "err": traceback.format_exc(limit=2)}


def build_jobs(zip_path, prefix, out_dir):
    """Group zip members into (study, series) work units."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    groups = collections.defaultdict(list)
    for n in names:
        if not n.startswith(prefix) or not n.endswith(".dcm"):
            continue
        parts = n.split("/")
        if len(parts) < 4:
            continue
        groups[(parts[1], parts[2])].append(n)
    return [(zip_path, st, se, sorted(ms), out_dir)
            for (st, se), ms in sorted(groups.items())]


def write_manifest(out_dir):
    """Record how this root was built, next to the volumes it describes.

    The root is the authority, not the training command: train.py loads .npy
    files it did not create, so a --pct-window flag there could disagree with
    the pixels and nothing would catch it. Reading it back off the root cannot.
    A root with no manifest is legacy and is (1.0, 99.0, 384, 48) by definition
    - every volume tree that existed before this file was written.
    """
    import json
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, MANIFEST), "w") as fh:
        json.dump({"lo_pct": LO_PCT, "hi_pct": HI_PCT,
                   "size": SIZE, "max_slices": MAX_SLICES}, fh)


def read_manifest(vol_root):
    """-> (lo_pct, hi_pct). Legacy roots have no manifest and are (1.0, 99.0).

    Follows one symlink level: the cotrain roots symlink to the competition and
    OAI trees, so the manifest lives on the target, not the combined root.
    """
    import json
    for probe in (vol_root, os.path.realpath(vol_root)):
        p = os.path.join(probe, MANIFEST)
        if os.path.exists(p):
            try:
                m = json.load(open(p))
                return float(m.get("lo_pct", 1.0)), float(m.get("hi_pct", 99.0))
            except Exception:
                break
    return 1.0, 99.0


def main():
    global SIZE, LO_PCT, HI_PCT
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/archive.zip")
    ap.add_argument("--prefix", default="train_series/")
    ap.add_argument("--out", default="data/volumes/train")
    ap.add_argument("--meta-out", default="data/processed/series_meta_train.parquet")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=SIZE,
                    help="In-plane output size. 384 is NOT throwing away real "
                         "detail, contrary to what this help said until "
                         "2026-08-12. That claim read the RECONSTRUCTED matrix "
                         "(median 512); measuring AcquisitionMatrix on 768 "
                         "series (96.1%% availability) gives a median ACQUIRED "
                         "320 frequency / 268 phase, i.e. the 512 is 1.67-2.0x "
                         "zero-fill interpolation. True acquired resolution is "
                         "0.469 mm/px on the fine axis and 0.620 on the coarse "
                         "one, against 0.417 mm/px stored at 384 - so we already "
                         "oversample by 1.16x/1.44x, and only 14.3%% of series "
                         "are coarser at 384 than their own fine axis (0.5%% "
                         "than their coarse axis). c320 losing just 0.008 is the "
                         "same fact from the other side. 512 would buy real "
                         "detail on ~14%% of series for 1.8x storage and GPU. "
                         "Writes to a SEPARATE --out; the checkpoint records "
                         "size so the two never mix.")
    ap.add_argument("--lo-pct", type=float, default=LO_PCT)
    ap.add_argument("--hi-pct", type=float, default=HI_PCT,
                    help="Upper percentile of the intensity window. The clip "
                         "here is HARD, so the saturated voxel mass is pinned "
                         "at (100-hi) by construction - measured 1.000%% +/- "
                         "0.003%% over 250 volumes at the default. That makes "
                         "'how extensive is the very bright fluid' a "
                         "zero-information quantity, and scale-free statistics "
                         "of the clipped tail carry +0.09 AUC on Effusion under "
                         "scanner-grouped CV. Raising this is a NEW INPUT "
                         "CONTRACT: write a separate --out and expect every "
                         "existing checkpoint to be incompatible with it.")
    args = ap.parse_args()

    # process_series reads the module-level SIZE. Workers are forked, so
    # rebinding it here - before the pool exists - is what actually reaches
    # them; passing it per-job would mean touching the job tuple everywhere.
    SIZE = args.size
    LO_PCT, HI_PCT = args.lo_pct, args.hi_pct
    if args.size != 384 and args.out.rstrip("/").endswith("train"):
        raise SystemExit(
            f"--size {args.size} would overwrite the 384 volumes at {args.out}. "
            "Pass a distinct --out (e.g. data/volumes/train512).")
    if (LO_PCT, HI_PCT) != (1.0, 99.0) and args.out.rstrip("/").endswith("train"):
        raise SystemExit(
            f"--lo-pct/--hi-pct {LO_PCT}/{HI_PCT} would overwrite the canonical "
            f"1/99 volumes at {args.out} in place, silently invalidating every "
            f"existing checkpoint. Pass a distinct --out "
            f"(e.g. data/volumes/train_p998).")
    write_manifest(args.out)
    print(f"[preprocess] size {SIZE}, window p{LO_PCT}-p{HI_PCT}, "
          f"out {args.out}", flush=True)
    print(f"[preprocess] indexing {args.zip} ...", flush=True)
    jobs = build_jobs(args.zip, args.prefix, args.out)
    if args.limit:
        jobs = jobs[: args.limit]
    n_files = sum(len(j[3]) for j in jobs)
    print(f"[preprocess] {len(jobs)} series / {n_files} dicoms, "
          f"{args.workers} workers", flush=True)

    rows, done, bad = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(process_series, jobs, chunksize=4):
            rows.append(r)
            done += 1
            if r["status"] not in ("ok", "cached"):
                bad += 1
                if bad <= 10:
                    print(f"  [{r['status']}] {r['study'][-8:]}/{r['series'][-8:]} "
                          f"{r.get('err','')[:200]}", flush=True)
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} ({bad} bad)", flush=True)

    print(f"[preprocess] done {done}, {bad} failed", flush=True)
    try:
        import pandas as pd
        os.makedirs(os.path.dirname(args.meta_out), exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.meta_out)
        print(f"[preprocess] meta -> {args.meta_out}", flush=True)
    except Exception as e:
        print("meta write failed:", e)


if __name__ == "__main__":
    sys.exit(main())
