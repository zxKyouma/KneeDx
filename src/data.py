"""Study-level dataset over preprocessed knee-MRI volumes.

A study is 3-14 series across three planes and two contrast types. Different
findings live in different series - a meniscal tear reads best on sagittal, MCL
on coronal, effusion on fluid-sensitive - so the unit of prediction is the
study and the model has to see several series at once.

Series selection is deterministic at eval time and lightly randomised during
training, always preferring one series per (plane, fluid-sensitive) type before
doubling up on any type. That keeps the sampled subset spanning the joint
rather than collapsing onto whichever type happens to be most numerous.
"""

import json
import hashlib
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA", "Effusion",
            "Synovitis", "Baker's", "Contusion", "Fracture"]


def load_label_repair_sidecar(path, labels_path, gold_uids=(),
                              allow_unpromoted=False, promotion_path=None):
    """Load and validate training-label repairs from a JSONL sidecar.

    Each row names one ``(uid, finding)`` cell and exactly one action:
    ``target`` with ``target``, ``weight`` with ``weight``, or ``interval``
    with either ``lo``/``hi`` or ``interval: [lo, hi]``. Provenance is
    mandatory even though it does not affect the loss: ``source_grade`` must
    match the grade in ``labels_path``, ``sol_call`` must be non-null, and
    either numeric ``sol_confidence`` or descriptive string
    ``sol_confidence_rating`` must be supplied. Protocol metadata must be a
    JSON object under ``protocol`` or ``protocol_metadata``.

    The returned repair map is keyed by UID and finding index. It is still an
    all-roster map; callers must filter it against the post-split training UID
    set before handing it to a dataset.
    """
    base_rows = {}
    with open(labels_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{labels_path}:{line_no}: invalid JSON: {e}") from e
            uid = row.get("uid")
            if not isinstance(uid, str) or not uid:
                raise ValueError(f"{labels_path}:{line_no}: missing/invalid uid")
            if uid in base_rows:
                raise ValueError(f"{labels_path}:{line_no}: duplicate base-label uid {uid!r}")
            base_rows[uid] = row

    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    promotion = None
    if promotion_path:
        with open(promotion_path, encoding="utf-8") as handle:
            promotion = json.load(handle)
        if (not isinstance(promotion, dict) or
                promotion.get("program") != "direct_sol_expansion_v2_promotion" or
                promotion.get("promoted_sidecar_sha256") != digest):
            raise ValueError(
                f"{promotion_path}: promotion receipt does not authorize this exact sidecar")
    repairs = {}
    seen = set()
    action_counts = {"target": 0, "weight": 0, "interval": 0}
    gold_uids = set(gold_uids)

    for line_no, raw_line in enumerate(raw.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: entry must be a JSON object")

        uid = row.get("uid")
        finding = row.get("finding")
        action = row.get("action")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"{path}:{line_no}: missing/invalid uid")
        if uid not in base_rows:
            raise ValueError(f"{path}:{line_no}: uid {uid!r} is absent from base labels")
        if uid in gold_uids:
            raise ValueError(f"{path}:{line_no}: uid {uid!r} overlaps Gold-58")
        if finding not in FINDINGS:
            raise ValueError(f"{path}:{line_no}: unknown finding {finding!r}")
        if action not in action_counts:
            raise ValueError(
                f"{path}:{line_no}: action must be target, weight, or interval")
        key = (uid, finding)
        if key in seen:
            raise ValueError(
                f"{path}:{line_no}: duplicate uid/finding {uid!r}/{finding!r}")
        seen.add(key)

        source_grade = row.get("source_grade")
        if isinstance(source_grade, bool) or not isinstance(source_grade, int):
            raise ValueError(f"{path}:{line_no}: source_grade must be an integer")
        grades = base_rows[uid].get("grades")
        if not isinstance(grades, dict):
            raise ValueError(
                f"{path}:{line_no}: uid {uid!r} has no base source grades")
        base_grade = grades.get(finding, -1)
        if source_grade != base_grade:
            raise ValueError(
                f"{path}:{line_no}: source_grade mismatch for {uid!r}/{finding!r}: "
                f"sidecar {source_grade}, base {base_grade}")
        if ("sol_call" not in row or row["sol_call"] is None
                or isinstance(row["sol_call"], (dict, list))):
            raise ValueError(f"{path}:{line_no}: sol_call must be a non-null scalar")
        confidence = row.get("sol_confidence")
        confidence_rating = row.get("sol_confidence_rating")
        if confidence is None and confidence_rating is None:
            raise ValueError(
                f"{path}:{line_no}: provide sol_confidence or descriptive "
                "sol_confidence_rating")
        if confidence is not None:
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0):
                raise ValueError(f"{path}:{line_no}: sol_confidence must be in [0,1]")
        if confidence_rating is not None and (
                not isinstance(confidence_rating, str) or not confidence_rating.strip()):
            raise ValueError(
                f"{path}:{line_no}: sol_confidence_rating must be a non-empty "
                "descriptive string")
        if "protocol" in row and "protocol_metadata" in row:
            raise ValueError(
                f"{path}:{line_no}: use protocol or protocol_metadata, not both")
        protocol = row.get("protocol_metadata", row.get("protocol"))
        if not isinstance(protocol, dict):
            raise ValueError(
                f"{path}:{line_no}: protocol metadata must be a JSON object")
        if protocol.get("name") == "direct_sol_repair_v2":
            status = protocol.get("status")
            if status == "column_only_assay_only":
                if not allow_unpromoted:
                    raise ValueError(
                        f"{path}:{line_no}: candidate repair requires the explicit "
                        "column-only assay flag or a passed promotion gate")
            elif status == "promoted":
                gate_hash = protocol.get("two_seed_training_gate_sha256")
                if not isinstance(gate_hash, str) or len(gate_hash) != 64:
                    raise ValueError(
                        f"{path}:{line_no}: promoted repair lacks its two-seed gate hash")
                target_name = {
                    "PF OA": "pf_oa", "Contusion": "contusion",
                    "Synovitis": "synovitis", "Lateral OA": "lateral_oa",
                }.get(finding)
                if (promotion is None or promotion.get("target") != target_name or
                        promotion.get("training_gate_sha256") != gate_hash):
                    raise ValueError(
                        f"{path}:{line_no}: promoted repair requires its matching "
                        "--label-repair-promotion receipt")
            else:
                raise ValueError(
                    f"{path}:{line_no}: direct_sol_repair_v2 has invalid status {status!r}")

        action_fields = {k for k in ("target", "weight", "interval", "lo", "hi")
                         if k in row}
        spec = {"action": action}
        if action == "target":
            if action_fields != {"target"}:
                raise ValueError(
                    f"{path}:{line_no}: target action requires only target")
            value = row["target"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{path}:{line_no}: target must be in [0,1]")
            spec["target"] = float(value)
        elif action == "weight":
            if action_fields != {"weight"}:
                raise ValueError(
                    f"{path}:{line_no}: weight action requires only weight")
            value = row["weight"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value < 0.0):
                raise ValueError(
                    f"{path}:{line_no}: weight must be finite and non-negative")
            spec["weight"] = float(value)
        else:
            if action_fields == {"interval"}:
                interval = row["interval"]
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    raise ValueError(
                        f"{path}:{line_no}: interval must be [lo, hi]")
                lo, hi = interval
            elif action_fields == {"lo", "hi"}:
                lo, hi = row["lo"], row["hi"]
            else:
                raise ValueError(
                    f"{path}:{line_no}: interval action requires only lo/hi "
                    f"or interval")
            if (isinstance(lo, bool) or isinstance(hi, bool)
                    or not isinstance(lo, (int, float))
                    or not isinstance(hi, (int, float))
                    or not np.isfinite(lo) or not np.isfinite(hi)
                    or not 0.0 <= lo <= hi <= 1.0):
                raise ValueError(
                    f"{path}:{line_no}: interval needs 0 <= lo <= hi <= 1")
            spec["lo"], spec["hi"] = float(lo), float(hi)

        repairs.setdefault(uid, {})[FINDINGS.index(finding)] = spec
        action_counts[action] += 1

    if not seen:
        raise ValueError(f"{path}: sidecar contains no entries")
    metadata = {
        "path": os.path.abspath(path),
        "sha256": digest,
        "entries": len(seen),
        "uids": len(repairs),
        "actions": action_counts,
        "promotion": os.path.abspath(promotion_path) if promotion_path else None,
    }
    return repairs, metadata


def training_label_repairs(repairs, train_uids, val_uids):
    """Select repairs for post-split training rows; validation is never read."""
    train_uids, val_uids = set(train_uids), set(val_uids)
    if train_uids & val_uids:
        raise ValueError("training and validation UID sets overlap")
    selected = {u: cells for u, cells in repairs.items() if u in train_uids}
    return selected, sum(len(v) for v in selected.values()), sum(
        len(cells) for u, cells in repairs.items() if u in val_uids)

# Pulse-sequence weighting, parsed from SeriesDescription.
#
# The 6-way (plane, fluid_sensitive) code conflates sequences that look nothing
# alike: a T1 coronal and a PD coronal, both without fat suppression, get the
# same token, yet T1 shows marrow and anatomy while PD shows menisci and
# cartilage. 82.3% of series carry a usable description and it splits cleanly -
# PD 46%, T2 24%, T1 21%. This recovers that axis.
#
# Read from DICOM at inference, so the notebook MUST use this exact function.
# Order matters: STIR is inversion-recovery and would otherwise be caught by
# the T2 pattern, and "PDW"/"DP" (densidad protonica, densite de protons - the
# corpus is nine languages) must be caught before the generic checks.
SEQ_CLASSES = ["unknown", "PD", "T1", "T2", "STIR"]
SEQ_ID = {s: i for i, s in enumerate(SEQ_CLASSES)}


def parse_sequence(desc):
    """SeriesDescription -> index into SEQ_CLASSES. 0 when unreadable.

    Deliberately conservative: an ambiguous description returns "unknown"
    rather than a guess, because a wrong sequence token is worse than a missing
    one - the model can learn that "unknown" is uninformative, but it cannot
    learn that a label is a lie.
    """
    if not desc:
        return 0
    d = str(desc).lower()
    if "dummy" in d:
        return 0
    # \b is useless here: regex counts "_" as a word character, so \bpdw?\b
    # fails on "PDW_TSE_Sag" - which is 1,100 series on its own. Use explicit
    # letter lookarounds instead, which treat "_", " ", "." and digits as
    # separators the way these descriptions actually do.
    L = r"(?<![a-z])"
    R = r"(?![a-z])"
    # Inversion recovery first: STIR/TIRM are fluid-sensitive but are not what
    # anyone means by T2, and often carry a t2 token too. "spir"/"spair" are
    # spectral fat saturation, NOT inversion recovery, so they must not match.
    if re.search(r"stir|tirm", d):
        return SEQ_ID["STIR"]
    # Proton density, including the "W"-for-weighted suffix, the plane-prefixed
    # form (CORPD FSE FS), the nine-language spellings, and intermediate-
    # weighted TSE - the standard knee workhorse, read as PD in practice.
    if re.search(rf"{L}pdw?{R}|(?:cor|sag|tra|ax)pd{R}|proton|densid|densite"
                 rf"|{L}dp{R}|{L}int{R}", d):
        return SEQ_ID["PD"]
    if re.search(rf"{L}t1w?{R}", d):
        return SEQ_ID["T1"]
    if re.search(rf"{L}t2w?{R}", d):
        return SEQ_ID["T2"]
    return 0


PLANES = ["Sagittal", "Coronal", "Axial"]
# (plane, fluid_sensitive) -> id. Fluid_Sensitive and Fat_Suppression are
# perfectly correlated in this dataset, so one bit covers both.
SERIES_TYPES = [(p, f) for p in PLANES for f in (0, 1)]
STYPE_ID = {t: i for i, t in enumerate(SERIES_TYPES)}
# --tfcrop-root extra series (scratch/tfcrop_20261011): ids 6.. after the 6
# standard types; EVMIL initialises their reliability prior from the
# (plane, fluid) in TFCROP_PRIOR_AS. Unused unless the flag is on.
TFCROP_KEYS = ["lat_fs", "med_fs", "lat_nf", "med_nf"]
TFCROP_TYPE = {k: len(SERIES_TYPES) + i for i, k in enumerate(TFCROP_KEYS)}
TFCROP_SWAP = {"lat_fs": "med_fs", "med_fs": "lat_fs", "lat_nf": "med_nf", "med_nf": "lat_nf"}
TFCROP_PRIOR_AS = [("Coronal", 1), ("Coronal", 1), ("Coronal", 0), ("Coronal", 0)]


def grade_to_prob(grades, table, silent=None):
    """Map ordinal report grades to soft binary targets.

    `silent` optionally overrides the grade -1 ("report does not mention it")
    cell per finding. Silence means very different things per finding: the
    report omits Synovitis 53% of the time and a quarter of those are positive,
    while an unmentioned ACL is negative essentially always. A single pooled
    value gets both wrong, and it gets them wrong on the findings where the -1
    cell holds the most studies.
    """
    out = []
    for i, g in enumerate(grades):
        g = int(g)
        if g == -1 and silent is not None:
            out.append(float(silent[i]))
        else:
            out.append(table.get(g, 0.05))
    return np.array(out, dtype=np.float32)


TARGET_MM_PER_PX = 0.417        # corpus median for the stored 384 volumes


def load_sequence(path="data/processed/series_geom.json"):
    """{study/series: SEQ_CLASSES index} parsed from SeriesDescription.

    Empty dict if the file is missing, which disables the feature rather than
    failing - the same contract as load_geometry/load_laterality. Whether a run
    actually used it is recorded in the checkpoint, not inferred here.
    """
    try:
        out = {}
        for r in json.load(open(path)):
            out[f"{r['study']}/{r['series']}"] = parse_sequence(r.get("series_desc"))
        return out
    except Exception:
        return {}


def load_geometry(path="data/processed/series_geom.json"):
    """{study/series: effective mm-per-pixel of the stored 384 volume}.

    Volumes were all resized to 384 square regardless of acquisition FOV, so a
    lesion of a given physical size spans a different number of pixels depending
    on the scanner. The bulk of the corpus is tight (p90/p10 = 1.20x), but the
    tails run 0.183 to 0.833 mm/px, so this mostly rescues outlier acquisitions.
    """
    try:
        out = {}
        for r in json.load(open(path)):
            sp, rw = r.get("spacing", 0), r.get("rows", 0)
            if sp and rw:
                out[f"{r['study']}/{r['series']}"] = sp * rw / 384.0
        return out
    except Exception:
        return {}


def load_zspacing(path="data/processed/series_geom.json"):
    """{study/series: through-plane slice thickness in mm} (scratch/physgeo)."""
    try:
        return {f"{r['study']}/{r['series']}": float(r["thickness"])
                for r in json.load(open(path)) if r.get("thickness")}
    except Exception:
        return {}


def load_aspect(path="data/processed/series_geom.json"):
    """{study/series: rows/cols of the acquisition matrix} (scratch/physgeo).

    Volumes were resized to 384 square; a non-square matrix with square
    physical pixels is therefore stretched along its shorter axis by rows/cols.
    """
    try:
        return {f"{r['study']}/{r['series']}": float(r["rows"]) / float(r["cols"])
                for r in json.load(open(path)) if r.get("rows") and r.get("cols")}
    except Exception:
        return {}


def fix_aspect(v, rows_over_cols):
    """Undo the anisotropic stretch of a square resize: shrink the over-
    stretched axis back to physical proportion and pad symmetrically."""
    import cv2
    n, H, W = v.shape
    r = float(rows_over_cols)
    if r > 1.0:                      # rows > cols: width was stretched by r
        newW = max(8, int(round(W / r))); newH = H
    else:                            # cols > rows: height was stretched by 1/r
        newH = max(8, int(round(H * r))); newW = W
    out = np.stack([cv2.resize(sl, (newW, newH), interpolation=cv2.INTER_AREA)
                    for sl in v])
    ph, pw = H - newH, W - newW
    return np.pad(out, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))


def resample_to_scale(v, mm_per_px, target=TARGET_MM_PER_PX, size=None):
    """Resample so one pixel spans `target` mm, then centre crop/pad to `size`."""
    import cv2
    size = size or v.shape[-1]
    if not mm_per_px or abs(mm_per_px - target) / target < 0.02:
        return v
    new = int(round(v.shape[-1] * mm_per_px / target))
    if new < 8 or new > 4096:
        return v
    out = np.stack([cv2.resize(sl, (new, new),
                               interpolation=cv2.INTER_AREA if new < v.shape[-1]
                               else cv2.INTER_LINEAR) for sl in v])
    if new >= size:                                   # crop the centre
        o = (new - size) // 2
        return out[:, o:o + size, o:o + size]
    pad = size - new                                  # pad symmetrically
    a, b = pad // 2, pad - pad // 2
    return np.pad(out, ((0, 0), (a, b), (a, b)))


def near_duplicate_groups(texts, threshold=0.8, n_hash=128, bands=32, seed=0):
    """Cluster reports by near-duplicate text; returns {key: group_id}.

    Exact-hash grouping leaves 6-10% of each validation fold with a >=0.8
    Jaccard twin in its training half, and 2.3-5.4% with a byte-identical
    12-grade label vector reaching it through that twin. Institutional report
    templates differ by a sentence, so exact hashing sees them as distinct while
    the derived labels are the same.

    That matters twice: it inflates fold-AUC, which selects checkpoints, and it
    contaminates the out-of-fold predictions that self-training will consume - a
    study whose near-twin was in training gets a partly in-sample prediction.

    MinHash + LSH banding over character 5-grams, then union-find over verified
    pairs. Deterministic given `seed`, which matters because every fold must
    compute the identical clustering or the folds will overlap.
    """
    import hashlib

    keys = list(texts)
    rng = np.random.RandomState(seed)
    # 64-bit permutations for the standard (a*x + b) mod prime MinHash.
    P = (1 << 61) - 1
    A = rng.randint(1, P, size=n_hash).astype(np.uint64)
    B = rng.randint(0, P, size=n_hash).astype(np.uint64)

    sig = np.full((len(keys), n_hash), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, k in enumerate(keys):
        t = " ".join(str(texts[k]).lower().split())
        grams = {t[j:j + 5] for j in range(max(len(t) - 4, 1))}
        if not grams:
            continue
        h = np.array([int.from_bytes(hashlib.md5(g.encode()).digest()[:8], "little")
                      for g in grams], dtype=np.uint64) % np.uint64(P)
        # (a*x + b) mod P for every permutation, then the column-wise minimum.
        sig[i] = ((A[None, :] * h[:, None] + B[None, :]) % np.uint64(P)).min(axis=0)

    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    rows = n_hash // bands
    for band in range(bands):
        buckets = {}
        blk = sig[:, band * rows:(band + 1) * rows]
        for i in range(len(keys)):
            buckets.setdefault(blk[i].tobytes(), []).append(i)
        for idxs in buckets.values():
            if len(idxs) < 2 or len(idxs) > 200:      # skip degenerate buckets
                continue
            a = idxs[0]
            for b in idxs[1:]:
                # Verify with the full signature rather than trusting the band.
                if (sig[a] == sig[b]).mean() >= threshold:
                    union(a, b)
    return {k: find(i) for i, k in enumerate(keys)}


def load_laterality(path="data/processed/laterality_map_v2.json"):
    """{study: 'L'|'R'} and {study/series: image row axis}, or empty if absent.

    v2 is DICOM tag where present (55%) plus an image classifier elsewhere,
    giving 100% coverage. v1 reached only 68%, and did so partly by regexing
    train.csv's Report column - a signal test.csv does not carry, so 282 studies
    were flipped in training and not at inference.

    Coverage matters more than it looks: an unknown side leaves the study
    unflipped, which is wrong for the ~50% that are right knees. v1 therefore
    implied ~22% of the corpus in the wrong handedness; v2 implies ~1.6%.
    """
    try:
        d = json.load(open(path))
        return d.get("side", {}), d.get("row_axis", {})
    except Exception:
        return {}, {}


# Mirror-with-label-swap. A plain horizontal flip is wrong here and was removed
# for that reason; a flip that ALSO swaps the paired labels is a different
# operation and is what every top team with a left/right target pair used
# (RSNA 2025 5th measured +0.01; RSNA 2024 3rd said it "effectively doubled the
# amount of data"). It attacks our prevalence asymmetry head on - medial
# meniscus is 56.8% of the corpus against lateral's 27.2%, and after
# flip-with-swap both queries see identical data volume.
#
# FLIP_SWAP holds the index pairs that genuinely exchange under a mirror.
FLIP_SWAP = ((2, 3), (4, 5))          # Medial/Lateral Meniscus, Medial/Lateral OA
# These are laterality-SPECIFIC but have no counterpart among the twelve, so a
# mirrored study carries a label that no longer describes the image: a mirrored
# MCL sits where the LCL belongs, and a mirrored Baker's cyst is posterolateral.
# Their loss is masked on flipped studies rather than taught wrong. Everything
# else is central or side-agnostic (ACL, PF OA, Effusion, Synovitis, Contusion,
# Fracture) and needs neither swap nor mask.
FLIP_MASK = (1, 9)                    # MCL, Baker's
# Auxiliary report fields exchange too; LCL/PLC has the same unpaired problem.
AUX_FLIP_SWAP = ((0, 1),)             # Cartilage Medial / Cartilage Lateral

# --aux-seg-w (scratch/auxdice/PREREG.md). Mask files are
# <root>/<StudyUID>/<SeriesUID>_msk.npy, uint8 (Zraw, 384, 384), values
# 0..5 in the OAI-ZIB/OAIZIB-CM convention (scratch/structruth/unet2d.py
# CLASS_NAMES): 0 background, 1 femur_bone, 2 femoral_cartilage,
# 3 tibia_bone, 4 medial_tibial_cartilage, 5 lateral_tibial_cartilage.
# AUX_SEG_SIZE is the decoder's output resolution (AuxSegHead upsamples the
# backbone's stride-32 feature map by 4x: 12x12 -> 48x48 for
# convnext_tiny/384) - masks are nearest-resized down to this size in the
# dataset so the loss never has to resize per training step.
AUX_SEG_CLASSES = 6
AUX_SEG_SIZE = 48

# ---- MOAKS subregional auxiliary targets (--moaks-aux) ----------------------
# scratch/latoarescue/RUBRIC_MAP.md §3, frozen: 62 auxiliary columns built by
# src/build_moaks_subregion_targets.py in EXACTLY this order. Blocks: cartilage
# AREA per subregion, cartilage DEPTH per subregion, osteophyte per site, BML
# size per subregion, meniscal extrusion, compartment rubric-grade summary.
# Compartment assignment is the production one (oai_labels.CART/OSTEO,
# trochlea -> PF). Within the TF families the medial and lateral sublists are
# index-ALIGNED (element i of the medial list mirrors element i of the lateral
# list) - the flip-swap table below is built from that alignment plus the
# explicit patella/trochlea pairs, all verified against the NDA data
# dictionary text embedded in row 2 of oai_kmrisemiquant01.txt.
_MOAKS_CART = ["cmf", "pmf", "cmt", "pmt", "amt",     # medial TF (femur c/p, tibia c/p/a)
               "clf", "plf", "clt", "plt", "alt",     # lateral TF, aligned
               "lp", "mp", "alf", "amf"]              # PF: patella lat/med, trochlea lat/med
_MOAKS_OS = ["fmc", "fmp", "tm",                      # medial TF
             "flc", "flp", "tl",                      # lateral TF, aligned
             "pi", "pl", "pm", "ps", "fla", "fma"]    # PF
_MOAKS_BML = ["cmf", "pmf", "amt", "tmc", "pmt",      # medial TF
              "clf", "plf", "alt", "clt", "lpt",      # lateral TF, aligned
              "lp", "mp", "ltr", "mtr",               # PF
              "sst"]                                  # tibia sub-spinous (midline)
_MOAKS_EXT = ["mma", "mmm", "lma", "lml"]             # b_*_extrusion columns
_MOAKS_COMP = ["MedialOA", "LateralOA", "PFOA"]

MOAKS_AUX_BLOCKS = (
    ("area",  ["area_" + s for s in _MOAKS_CART]),
    ("depth", ["depth_" + s for s in _MOAKS_CART]),
    ("osteo", ["os_" + s for s in _MOAKS_OS]),
    ("bml",   ["bml_" + s for s in _MOAKS_BML]),
    ("extr",  ["extr_" + s for s in _MOAKS_EXT]),
    ("comp",  ["comp_" + s for s in _MOAKS_COMP]),
)
MOAKS_AUX_COLS = [c for _, _cols in MOAKS_AUX_BLOCKS for c in _cols]

# Side-symmetric columns: a mirrored knee maps them onto themselves.
# patella inferior/superior, tibia sub-spinous, and the PF compartment summary.
MOAKS_AUX_SELF = ("os_pi", "os_ps", "bml_sst", "comp_PFOA")


def _moaks_flip_pairs():
    """med<->lat mirror index pairs over MOAKS_AUX_COLS.

    Data-dictionary verification (oai_kmrisemiquant01.txt row 2): every raw
    column's description names its side and site explicitly, so each pair
    below is a literal medial<->lateral text mirror - including the two
    irregularly named BML tibial columns (tmc "tibia medial central" <-> clt
    "tibia lateral central"; pmt "tibia medial posterior" <-> lpt "tibia
    lateral posterior") and the extrusion directions (mmm "medial meniscus,
    medially" <-> lml "lateral meniscus, laterally"; mma <-> lma anteriorly).
    No column's mirror was ambiguous.
    """
    idx = {c: i for i, c in enumerate(MOAKS_AUX_COLS)}
    pairs = []
    add = lambda a, b: pairs.append((idx[a], idx[b]))
    for p in ("area", "depth"):                       # cartilage, both blocks
        for m_, l_ in zip(_MOAKS_CART[:5], _MOAKS_CART[5:10]):
            add(p + "_" + m_, p + "_" + l_)
        add(p + "_mp", p + "_lp")                     # patella medial/lateral
        add(p + "_amf", p + "_alf")                   # trochlea medial/lateral
    for m_, l_ in zip(_MOAKS_OS[:3], _MOAKS_OS[3:6]):
        add("os_" + m_, "os_" + l_)
    add("os_pm", "os_pl")
    add("os_fma", "os_fla")
    for m_, l_ in zip(_MOAKS_BML[:5], _MOAKS_BML[5:10]):
        add("bml_" + m_, "bml_" + l_)
    add("bml_mp", "bml_lp")
    add("bml_mtr", "bml_ltr")
    add("extr_mma", "extr_lma")
    add("extr_mmm", "extr_lml")
    add("comp_MedialOA", "comp_LateralOA")
    return tuple(pairs)


MOAKS_AUX_FLIP_SWAP = _moaks_flip_pairs()
# The table must be a full partition: every column is either in exactly one
# pair or explicitly side-symmetric. A silent gap here would anti-teach the
# med/lat distinction the program targets, so it is asserted at import.
_ms = [i for a, b in MOAKS_AUX_FLIP_SWAP for i in (a, b)]
assert len(_ms) == len(set(_ms)), "MOAKS_AUX_FLIP_SWAP has a duplicated column"
assert (sorted(_ms + [MOAKS_AUX_COLS.index(c) for c in MOAKS_AUX_SELF])
        == list(range(len(MOAKS_AUX_COLS)))), \
    "MOAKS_AUX_FLIP_SWAP + MOAKS_AUX_SELF do not partition the 62 columns"
del _ms


def load_moaks_aux(path, table=None):
    """oai_subregion_targets.jsonl -> ({uid: np.array(62)}, {uid: meta}).

    Grades map through the SAME production ladder as every other supervision
    source (the oai_labels.py docstring rule - no second calibration). A null
    grade means "unscored" and becomes NaN, which the trainer's per-cell mask
    turns into exactly zero gradient - masked, never imputed (the osteophyte
    columns are unscored for ~77% of knees, and a fabricated soft negative
    there would be the cotrain-val trap in aux form).

    Unlike load_aux_labels this FAILS LOUD: the sidecar is a decision-gating
    input, and a malformed row must kill the run, not silently disable the
    head. Every row must carry all 62 columns explicitly (absent != null:
    absence would alias onto the -1 "report silent" ladder cell, which is a
    real 0.10 target, not a mask).
    """
    table = PRIOR_TABLE if table is None else table
    out, meta = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            g = r["grades"]
            missing = [c for c in MOAKS_AUX_COLS if c not in g]
            if missing:
                raise ValueError(
                    f"{path}: uid {r['uid']} lacks columns {missing[:5]} "
                    f"(+{max(len(missing) - 5, 0)} more) - sidecar and "
                    f"MOAKS_AUX_COLS disagree")
            out[r["uid"]] = np.array(
                [np.nan if g[c] is None else table[int(g[c])]
                 for c in MOAKS_AUX_COLS], np.float32)
            meta[r["uid"]] = {k: r[k] for k in
                              ("oa_agree", "oa_floor_pos", "oa_rubric_pos")
                              if k in r}
    return out, meta

# Shared default for the coverage lookup. Module-level so the per-item path
# does not allocate a 12-vector for every competition study on every epoch.
_FW_ONES = np.ones(len(FINDINGS), np.float32)


def _domain_rand_series(v):
    """Acquisition-style randomization for one series (float32, [0,1]).

    Goal per the site-shift program: destroy the usefulness of ACQUISITION
    STYLE as a feature while preserving anatomy. Six ops, each a caricature
    of a real scanner/protocol difference; 2-3 are applied per call. This is
    deliberately more violent than --vendor-aug (gamma/gain/noise): that arm
    measured ~null, and the diagnosis was that mild intensity jitter does not
    span reconstruction differences. Fourier-domain ops do.
    """
    import random as _r
    S, H, W = v.shape
    ops = _r.sample(range(6), _r.choice((2, 3)))

    if 0 in ops:                      # radial amplitude warp (recon filter)
        fy = np.fft.fftfreq(H)[:, None]
        fx = np.fft.fftfreq(W)[None, :]
        r = np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5)
        gain = np.exp(_r.uniform(-0.8, 0.8) * (r - r.mean())).astype(np.float32)
        F = np.fft.fft2(v)
        v = np.real(np.fft.ifft2(F * gain[None])).astype(np.float32)
    if 1 in ops:                      # bias field / coil shading
        g = np.random.normal(size=(4, 4)).astype(np.float32)
        import cv2
        field = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        field = 1.0 + _r.uniform(0.1, 0.35) * (field / (np.abs(field).max() + 1e-6))
        v = v * field[None]
    if 2 in ops:                      # Rician-ish noise (low-SNR coil)
        s = _r.uniform(0.01, 0.05)
        n1 = np.random.normal(0, s, v.shape).astype(np.float32)
        n2 = np.random.normal(0, s, v.shape).astype(np.float32)
        v = np.sqrt(np.clip(v + n1, 0, None) ** 2 + n2 ** 2)
    if 3 in ops:                      # anisotropic k-space truncation + Gibbs
        fy = _r.uniform(0.45, 1.0)
        fx = _r.uniform(0.45, 1.0)
        F = np.fft.fftshift(np.fft.fft2(v), axes=(-2, -1))
        my, mx = int(H * fy / 2), int(W * fx / 2)
        M = np.zeros((H, W), np.float32)
        M[H // 2 - my:H // 2 + my, W // 2 - mx:W // 2 + mx] = 1.0
        v = np.real(np.fft.ifft2(np.fft.ifftshift(F * M[None],
                                                  axes=(-2, -1)))).astype(np.float32)
    if 4 in ops:                      # monotone contrast warp (vendor curves)
        xs = np.linspace(0, 1, 6)
        ys = np.sort(np.clip(xs + np.random.normal(0, 0.12, 6), 0, 1))
        ys[0], ys[-1] = 0.0, 1.0
        v = np.interp(np.clip(v, 0, 1), xs, ys).astype(np.float32)
    if 5 in ops:                      # mild ghosting (motion/pulsation)
        a = _r.uniform(0.03, 0.12)
        ax = _r.choice((1, 2))
        v = (1 - a) * v + a * np.roll(v, _r.randrange(8, max(H // 4, 9)), axis=ax)
    return np.clip(v, 0, 1).astype(np.float32)


class KneeStudyDataset(Dataset):
    TFCROP_ROOT = ""  # --tfcrop-root, set from args/ck['args']; "" = off
    def __init__(self, studies, series_index, labels=None, vol_root="data/volumes/train",
                 n_series=6, n_slices=24, size=384, train=True,
                 laterality=None, geometry=None, sequence=None,
                 aux_labels=None, agree=None, flip_swap=0.0, cutmix=0.0,
                 cutmix_alpha=1.0, coverage=None, vendor_aug=0.0,
                 series_dropout=0.0, spatial_aug=0.0, pairmax=None,
                 domain_rand=0.0, aux_flip_swap=None, aux_seg_root="",
                 pfrel_series=False, bone_series=False, bone_fluid=1,
                 native_ctx=False, slice_ctx=3, affine_p=0.7,
                 pick_best_twin=False, series_quality=None,
                 ctx_mm=0.0, zspacing=None, aspect=None,
                 stratified_slices=False, full_fov=False, label_repair=None,
                 evidence_intervals=None, evidence_sampling=False,
                 xfuse_pairs=None):
        """
        studies      : list of StudyInstanceUID
        series_index : {study: [(series_uid, plane, fluid_sensitive), ...]}
        labels       : {study: np.array(12,) float soft targets} or None
        coverage     : {study: np.array(12,) in [0,1]} or None. Per-study,
                       per-finding loss weight, multiplied into `fw` alongside
                       the flip mask. It exists for co-training on an EXTERNAL
                       pool whose labels cover only some of the twelve: an OAI
                       study scored with MOAKS supervises Synovitis and the
                       compartment-specific OA/meniscus items and must be
                       silent on Fracture and Contusion, which MOAKS does not
                       score. Weight 0 is exactly no gradient (already verified
                       for the --mute-after path, which multiplies the same
                       tensor), so a missing finding costs nothing rather than
                       teaching a fabricated negative.
                       Absent studies default to all-ones, so an external pool
                       can be added without touching the competition studies.
        """
        self.studies = studies
        self.series_index = series_index
        self.labels = labels
        # Repairs are a training-only contract. Even if a caller accidentally
        # passes the map to an evaluation dataset, discard it here so held-out
        # targets and batch keys remain byte-for-byte on the old path.
        self.label_repair = (label_repair or {}) if train else {}
        self.coverage = coverage
        self.vol_root = vol_root
        self.n_series = n_series
        # --tfcrop-root (scratch/tfcrop_20261011/PREREG.md): native-resolution
        # lateral/medial tibiofemoral coronal crops APPENDED as TFCROP_SLOTS extra
        # series after the n_series standard slots. Set by train.py / exporters
        # from ck["args"]["tfcrop_root"] via the class attribute. "" -> off: no
        # extra slots, every array and the RNG stream are byte-identical.
        self.tfcrop_root = str(KneeStudyDataset.TFCROP_ROOT or "")
        self.n_slices = n_slices
        self.size = size
        self.train = train
        # --native-ctx (scratch/nativectx): emit (n_slices, slice_ctx, H, W)
        # windows whose channels are the centre slice's immediate NATIVE
        # neighbours in the cached (48-cap) volume, instead of (n_slices,
        # H, W) for KneeNet._stack_ctx to pair up neighbours of the 24-
        # stack. Off (default) leaves every code path byte-identical.
        # scratch/pfrel: --pfrel serves ONE axial fluid-sensitive series and
        # the choice must be deterministic and IDENTICAL train/serve (the
        # relational read is about where structures sit inside one series; a
        # resampled series is a different study to this model). Selection is
        # by sorted SeriesInstanceUID, never by random.choice, and never
        # reshuffled by self.train.
        #
        # Coverage measured 2026-09-16 on data/raw/train_series.csv x
        # report_labels_v2.jsonl: 4406/4406 labelled studies (100.0%) have at
        # least one Axial + Fluid_Sensitive series, so the two fallback rungs
        # below are dead code on this corpus. They exist because the HIDDEN
        # test set is not measured: rung 1 = best available axial series of any
        # type, rung 2 = no axial at all -> the study is dropped at train time
        # and the server must emit a zero logit (the model's own constant-zero
        # column) rather than guess.
        self.pfrel_series = bool(pfrel_series)
        # scratch/bonewsrpn: --bonewsrpn serves ONE sagittal fluid-sensitive
        # and ONE coronal fluid-sensitive series, chosen DETERMINISTICALLY
        # (lowest SeriesInstanceUID within the plane), identical at train and
        # serve. Fallback ladder, measured coverage in the PREREG: one plane
        # missing -> that plane's slot is simply absent (mask 0, objects from
        # the available plane only); BOTH missing -> no series at all, which
        # the model turns into an exact zero logit pair. Both rungs increment
        # bone_fallback; the count is reported, never silently absorbed.
        self.bone_series = bool(bone_series)
        # scratch/bonefam (--bone-series-fluid nonfs): the SAME deterministic
        # sag/cor rule over NON-fluid-sensitive series (T1/PD, Fluid_Sensitive
        # == 0) instead of FS. Default 1 = FS = byte-identical to bonewsrpn.
        if int(bone_fluid) not in (0, 1):
            raise ValueError(f"bone_fluid must be 0 or 1, got {bone_fluid!r}")
        if int(bone_fluid) != 1 and not self.bone_series:
            raise ValueError("bone_fluid=0 is only defined for bone_series")
        self.bone_fluid = int(bone_fluid)
        self.bone_fallback = 0
        self.pfrel_fallback = 0
        self.native_ctx = bool(native_ctx)
        # --pick-best-twin (scratch/twinserve): SERVE-TIME data-path flag, never
        # set by train.py. Default OFF -> _pick_series is byte-identical.
        self.pick_best_twin = bool(pick_best_twin)
        self.series_quality = series_quality or {}
        self.slice_ctx = int(slice_ctx)
        self.affine_p = float(affine_p)
        # scratch/physgeo: ctx_mm > 0 places the context channels at a fixed
        # PHYSICAL depth (+-ctx_mm, rounded to whole native slices, min 1)
        # using per-series slice thickness; aspect restores the acquisition
        # matrix proportion. Both default off -> byte-identical loader.
        self.ctx_mm = float(ctx_mm or 0.0)
        self.zspacing = zspacing or {}
        self.aspect = aspect or {}
        # scratch/ffov: sub-384 members (224/256-px ViTs, MRI-CORE) CROP the
        # 384-square cache to `size` (random in train, centre at eval) and so
        # see 34-44% of the field of view. full_fov=True resizes the whole
        # frame to size x size (INTER_AREA, same family as fix_aspect) with
        # no crop, train and eval alike. Default False = byte-identical
        # loader (the branch is not entered). Stored in ck['args']; every
        # consumer (export/eval/serve) must read it from there.
        self.full_fov = bool(full_fov)
        # scratch/stratwin: when enabled for TRAINING only, draw one slice
        # centre independently and uniformly inside each disjoint depth bin.
        # The default is deliberately off: setting the attribute consumes no
        # RNG and the old single-grid-jitter branch below remains untouched.
        # Evaluation never uses this sampler, even if a caller accidentally
        # passes the flag on an eval dataset.
        self.stratified_slices = bool(stratified_slices) and bool(train)
        # Evidence-aware centres are an opt-in training-only continuation arm.
        # The map is native-cache verified before construction; evaluation is
        # deliberately standard even if a caller passes the flag accidentally.
        self.evidence_intervals = evidence_intervals or {}
        self.evidence_sampling = bool(evidence_sampling) and bool(train)
        # Four of the twelve findings are medial/lateral. Left and right knees
        # are mirror images, so on coronal and axial slices - where the image
        # x-axis runs along the patient's left-right axis - "medial" sits on
        # opposite sides of the frame depending on the knee. Mirroring the right
        # knees puts medial on a consistent side and makes those findings
        # learnable; a random horizontal flip does the exact opposite, so it is
        # gone (the fibula is always lateral, so flipping teaches the wrong map).
        self.side, self.row_axis = laterality if laterality is not None else (({}, {}))
        self.geom = geometry if geometry is not None else {}
        # {study/series: SEQ_CLASSES index}. Empty means the feature is off and
        # every series reports 0 ("unknown"), which is what a model built
        # without the sequence embedding expects anyway.
        self.seq = sequence if sequence is not None else {}
        # Auxiliary report targets, present only for training studies. A study
        # without them is skipped by the aux loss rather than given zeros -
        # zeros would read as "explicitly normal" for all eight fields.
        self.aux_labels = aux_labels if aux_labels is not None else {}
        # Which mirror table the auxiliary target vector uses under
        # flip-with-swap. Default None keeps the module-level AUX_FLIP_SWAP
        # (report-aux convention, byte-identical to every prior run);
        # --moaks-aux passes MOAKS_AUX_FLIP_SWAP, whose 62-column mirror is
        # asserted complete at import.
        self.aux_flip_swap = (AUX_FLIP_SWAP if aux_flip_swap is None
                              else tuple(aux_flip_swap))
        self.pairmax = pairmax if pairmax is not None else {}
        self.agree = agree if agree is not None else {}
        # Probability of mirroring a TRAINING study and swapping its paired
        # labels. Decided once per study, never per series: the labels are
        # study-level, so mirroring only some of a study's series would leave
        # the swapped label describing none of them.
        self.flip_swap = float(flip_swap) if train else 0.0
        self.vendor_aug = float(vendor_aug) if train else 0.0
        self.domain_rand = float(domain_rand) if train else 0.0
        self.series_dropout = float(series_dropout) if train else 0.0
        self.spatial_aug = float(spatial_aug) if train else 0.0
        self.cutmix = float(cutmix) if train else 0.0
        self.cutmix_alpha = float(cutmix_alpha)
        # --aux-seg-w (scratch/auxdice/PREREG.md): "" (default) means this
        # dataset never looks at a mask path, never opens cv2, never
        # allocates the seg/seg_valid arrays - the __getitem__ path below is
        # gated on this exact attribute being truthy, so an unset flag is
        # byte-identical to before this feature existed.
        self.aux_seg_root = aux_seg_root or ""
        self._mix_uid = None
        # --xfuse (scratch/xfuse/PREREG.md): same-voxel FS/non-FS early fusion.
        # xfuse_pairs = load_xfuse_pairs(...) {study/FS_series: (partner uid,
        # frac (z_cached,), M (2,3))}. Each series' volume becomes (2*n_slices,
        # H, W): rows [0, n) are the series itself exactly as before, rows
        # [n, 2n) its geometry-matched partner resampled onto the SAME cached
        # slice positions and in-plane grid (zero where there is no partner /
        # the slice falls outside the partner slab). Every series-level op
        # below acts on both halves with one set of random params. None
        # (default) = attribute False, every branch below is skipped and the
        # loader is byte-identical.
        self.xfuse_pairs = xfuse_pairs
        self.xfuse = xfuse_pairs is not None
        if self.xfuse and (self.native_ctx or self.pfrel_series or self.bone_series
                           or self.aux_seg_root or self.ctx_mm > 0):
            raise ValueError("--xfuse is not combined with native_ctx/ctx_mm/"
                             "pfrel/bonewsrpn/aux-seg (not implemented)")

    def _label_repair_arrays(self, study, target):
        """Return repaired target plus weight/interval vectors for one row."""
        target = target.copy()
        weight = np.ones(len(FINDINGS), np.float32)
        cells = self.label_repair.get(study, {})
        for col, spec in cells.items():
            if spec["action"] == "target":
                target[col] = np.float32(spec["target"])
            elif spec["action"] == "weight":
                weight[col] = np.float32(spec["weight"])
        # Point targets are degenerate intervals. This makes CutMix with one
        # interval endpoint well-defined as the convex combination of target
        # sets, while the mask below keeps ordinary cells on standard BCE.
        lo, hi = target.copy(), target.copy()
        interval = np.zeros(len(FINDINGS), dtype=np.bool_)
        for col, spec in cells.items():
            if spec["action"] == "interval":
                lo[col], hi[col] = np.float32(spec["lo"]), np.float32(spec["hi"])
                interval[col] = True
        return target, weight, lo, hi, interval

    def __len__(self):
        return len(self.studies)

    def _pick_series(self, study):
        avail = list(self.series_index.get(study, []))
        if not avail:
            return []
        if self.bone_series:
            out = []
            for pl in ("Sagittal", "Coronal"):
                fl = [x for x in avail if x[1] == pl and x[2] == self.bone_fluid]
                if fl:
                    out.append(sorted(fl, key=lambda x: x[0])[0])
            if len(out) < 2:
                self.bone_fallback += 1
            return out
        if self.pfrel_series:
            ax = [s for s in avail if s[1] == "Axial"]
            fl = [s for s in ax if s[2] == 1]
            pool = fl or ax
            if not pool:
                self.pfrel_fallback += 1
                return []
            if not fl:
                self.pfrel_fallback += 1
            return [sorted(pool, key=lambda s: s[0])[0]]
        by_type = {}
        for s in avail:
            by_type.setdefault(STYPE_ID[(s[1], s[2])], []).append(s)
        # STYPE_ID is plane*2 + fluid, so a plain sorted() walks
        # Sag/0, Sag/1, Cor/0, Cor/1, Ax/0, Ax/1 - and with n_series=4 the
        # round-robin stops before ever reaching Axial, silently dropping it
        # from 70% of studies at eval while training still sampled it 25% of
        # the time. Order plane-major so the first picks span all three planes.
        order = sorted(by_type, key=lambda k: (k % 2, k // 2))
        if self.train:
            random.shuffle(order)
            for k in by_type:
                random.shuffle(by_type[k])
        if self.pick_best_twin:
            # Frozen rule (scratch/twinserve/PREREG.md): among candidate series
            # of the SAME (plane, fluid) type, serve the best-quality one.
            # pop() below takes the LAST element, so order each group worst ->
            # best. sorted(..., reverse=True) is stable and does NOT reverse
            # ties, so equal-quality series keep their CSV order and the
            # existing tie-break (last-listed wins) survives exactly.
            # A group with any unknown-quality member is left untouched.
            for _k, _lst in by_type.items():
                if len(_lst) < 2:
                    continue
                _q = [self.series_quality.get(f"{study}/{_s[0]}") for _s in _lst]
                if any(x is None for x in _q):
                    continue
                by_type[_k] = [_s for _, _s in
                               sorted(zip(_q, _lst), key=lambda t: t[0], reverse=True)]
        # Round-robin across types so the subset spans the joint.
        picked, i = [], 0
        while len(picked) < self.n_series and any(by_type.values()):
            k = order[i % len(order)]
            if by_type[k]:
                picked.append(by_type[k].pop())
            i += 1
            if i > 200:
                break
        return picked[: self.n_series]

    def _load_series(self, study, series_uid, plane=None, flip=False,
                     want_mask=False, mask_path=None, vol_arr=None):
        """want_mask is False on every call site unless --aux-seg-w is on
        (see _build). When False, not one extra line below executes - this
        function is textually and behaviourally identical to before
        --aux-seg-w existed. When True, mask_path is either a real file (the
        companion 6-class mask, pushed through the
        pushed through the EXACT SAME idx-select / crop / side-canon /
        flip-swap / (p=0.7) rotation ops as `v`, reusing the same drawn
        random values (no extra call to `random.*`, so the RNG stream `v`
        alone would have consumed is unaffected either way), then nearest-
        resized to AUX_SEG_SIZE. Physical-scale resampling
        (--use-geometry) and --vendor-aug/--domain-rand/--spatial-aug are
        NOT mask-aligned (documented gap, not exercised by this launch:
        USE_GEOM/VENDORAUG/DOMRAND/SPATAUG are all unset) - if geometry
        resampling would have fired for this series, the mask is marked
        invalid rather than silently misaligned.
        Returns `v` alone when want_mask is False (unchanged contract);
        `(v, seg, seg_valid)` when want_mask is True - seg is
        (n_slices, AUX_SEG_SIZE, AUX_SEG_SIZE) uint8, seg_valid a bool
        (False, seg all-zero, whenever mask_path is None/missing/unusable).
        """
        path = os.path.join(self.vol_root, study, series_uid + ".npy")
        try:
            # vol_arr: --tfcrop-root only (scratch/tfcrop_20261011); None on every
            # other call site, so this is the unchanged np.load path.
            vol = vol_arr if vol_arr is not None else np.load(path, mmap_mode="r")
        except Exception:
            if want_mask:
                return (np.zeros(self._vshape(), np.float32),
                        np.zeros((self.n_slices, AUX_SEG_SIZE, AUX_SEG_SIZE), np.uint8),
                        False)
            return np.zeros(self._vshape(), np.float32)

        mvol = None
        if want_mask and mask_path is not None:
            try:
                mvol = np.load(mask_path, mmap_mode="r")
                if mvol.shape[0] != vol.shape[0]:
                    mvol = None       # z mismatch: raw mask not aligned to this .npy
            except Exception:
                mvol = None

        z = vol.shape[0]
        # Keep z == n_slices on the legacy path too: it is the full cached
        # volume, and preserving that case makes the opt-in sampler a true
        # training-view augmentation rather than a silent coverage change.
        if z > self.n_slices:
            if self.train:
                if self.stratified_slices and z > self.n_slices:
                    idx = self._stratified_slice_indices(z)
                else:
                    # Jitter the sampling grid so the model sees varied slice sets.
                    # This branch is intentionally unchanged when the new flag
                    # is off: same draw, same rounding, same output bytes.
                    off = random.uniform(-0.5, 0.5)
                    idx = np.clip(np.linspace(0, z - 1, self.n_slices) + off, 0, z - 1)
            else:
                idx = np.linspace(0, z - 1, self.n_slices)
            idx = idx.round().astype(int)
            if self.evidence_sampling:
                intervals = self.evidence_intervals.get(f"{study}/{series_uid}")
                # Only accepted-study/verified-support series receive the
                # continuation intervention. Unrelated studies and series
                # retain the exact standard centres, preserving the matched
                # baseline path. Trainer preflight separately requires every
                # accepted UID to have at least one mapped support series.
                if intervals is not None:
                    idx = evidence_slice_indices(z, self.n_slices, idx, intervals)
        else:
            idx = np.clip(np.arange(self.n_slices) - (self.n_slices - z) // 2, 0, z - 1)

        if self.native_ctx:
            # (n_slices*slice_ctx, H, W): window i occupies rows i*k..i*k+k-1,
            # channel c = native index clip(idx[i] + c - k//2). Every series-
            # level op below acts on this flat stack exactly as it acts on
            # the plain stack, so all channels of a window share one set of
            # random params; it is folded back to (n, k, H, W) at the end.
            v = np.asarray(vol[self._ctx_index(idx, z, self._zstep(study, series_uid)).reshape(-1)]).astype(np.float32) / 255.0
        else:
            v = np.asarray(vol[idx]).astype(np.float32) / 255.0
        if self.xfuse:
            _xp, _xvalid = self._xfuse_partner(study, series_uid, idx, v.shape[1:])
            v = np.concatenate([v, _xp], 0)
        sm = np.asarray(mvol[idx]) if mvol is not None else None  # (n_slices,H,W) uint8

        # Normalise physical scale so a given lesion spans the same pixel count
        # regardless of acquisition FOV.
        mm = self.geom.get(f"{study}/{series_uid}")
        if mm:
            v = resample_to_scale(v, mm, size=v.shape[-1])
            sm = None      # not mask-aligned (docstring) - drop rather than misalign
        if self.aspect:
            r = self.aspect.get(f"{study}/{series_uid}")
            if r and abs(r - 1.0) > 0.02:
                v = fix_aspect(v, r)
                sm = None

        # Spatial crop: random during training, centre at eval.
        H, W = v.shape[1], v.shape[2]
        if self.size < H and self.full_fov:
            # scratch/ffov: whole field of view, downsampled, no crop.
            import cv2
            v = np.stack([cv2.resize(np.ascontiguousarray(sl), (self.size, self.size),
                                     interpolation=cv2.INTER_AREA) for sl in v])
            if sm is not None:
                sm = np.stack([cv2.resize(np.ascontiguousarray(sl), (self.size, self.size),
                                          interpolation=cv2.INTER_NEAREST) for sl in sm])
        elif self.size < H:
            crop_m = H - self.size
            if self.train:
                y0, x0 = random.randint(0, crop_m), random.randint(0, crop_m)
            else:
                y0 = x0 = crop_m // 2
            v = v[:, y0:y0 + self.size, x0:x0 + self.size]
            if sm is not None:
                sm = sm[:, y0:y0 + self.size, x0:x0 + self.size]

        # Canonicalise handedness instead of randomising it. Coronal and axial
        # mirror left-right in the image plane. Sagittal does not - but its
        # SLICE ORDER encodes the side: the slice normal is the patient's
        # left-right axis either way, so the stack runs medial-to-lateral in
        # opposite directions for a left vs a right knee. Reverse the stack
        # rather than the frame.
        if self.side.get(study) == "R":
            if plane in ("Coronal", "Axial"):
                v = v[:, :, ::-1]
                if sm is not None:
                    sm = sm[:, :, ::-1]
            elif plane == "Sagittal":
                v = self._rev_stack(v)
                if self.xfuse:
                    _xvalid = _xvalid[::-1]
                if sm is not None:
                    sm = sm[::-1]

        # Mirror-with-label-swap, applied AFTER canonicalisation so it is a true
        # reflection of an already-consistent knee rather than an undo of the
        # canonicalisation for half the corpus. Same axis convention as above:
        # coronal and axial mirror in-plane, sagittal reverses the slice stack,
        # because there the medial-lateral axis IS the stack.
        if flip:
            if plane in ("Coronal", "Axial"):
                v = v[:, :, ::-1]
                if sm is not None:
                    sm = sm[:, :, ::-1]
            elif plane == "Sagittal":
                v = self._rev_stack(v)
                if self.xfuse:
                    _xvalid = _xvalid[::-1]
                if sm is not None:
                    sm = sm[::-1]

        if self.train:
            # Full-frame training removed the random crop, which had been the
            # only spatial augmentation. Rotation/scale/shift preserve chirality
            # (unlike a horizontal flip) and are applied identically to every
            # slice of a series so the through-plane geometry stays consistent.
            if self.affine_p > 0 and random.random() < self.affine_p:
                import cv2
                ang = random.uniform(-10, 10)
                sc = random.uniform(0.92, 1.08)
                tx = random.uniform(-0.04, 0.04) * self.size
                ty = random.uniform(-0.04, 0.04) * self.size
                M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, sc)
                M[0, 2] += tx
                M[1, 2] += ty
                v = np.stack([cv2.warpAffine(sl, M, (self.size, self.size),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT)
                              for sl in v])
                # --aux-seg-w: same M, nearest interpolation (mask is a class
                # id map, not an intensity), applied to keep the decoder's
                # target aligned with what the classifier trunk actually sees.
                if sm is not None:
                    sm = np.stack([cv2.warpAffine(
                        np.ascontiguousarray(sl), M, (self.size, self.size),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                        for sl in sm])
            if random.random() < 0.3:                      # mild intensity jitter
                v = np.clip(v * random.uniform(0.85, 1.15) +
                            random.uniform(-0.06, 0.06), 0, 1)
            # Vendor-spanning intensity augmentation (--vendor-aug): the
            # site gap is largely an intensity/contrast phenomenon and the
            # policy above is intensity-mild. Gamma warps the contrast
            # curve (the main vendor axis), wider gain/offset spans
            # windowing differences, and Gaussian noise spans coil/field
            # SNR. Applied per SERIES - vendor effects are series-level.
            if self.vendor_aug > 0 and random.random() < self.vendor_aug:
                g = float(np.exp(random.uniform(-0.35, 0.35)))
                v = np.power(np.clip(v, 0, 1), g)
                v = v * random.uniform(0.7, 1.3) + random.uniform(-0.12, 0.12)
                if random.random() < 0.5:
                    sd = random.uniform(0.005, 0.03)
                    if self.native_ctx:
                        # Noise the 24-STACK (one field per centre slice), then
                        # window it - what the off path + _stack_ctx produce -
                        # rather than independent noise per channel.
                        nz = np.random.normal(0.0, sd, (self.n_slices,) + v.shape[1:])
                        v = v + nz.astype(np.float32)[self._stack_nbr().reshape(-1)]
                    else:
                        v = v + np.random.normal(0.0, sd, v.shape).astype(np.float32)
                v = np.clip(v, 0, 1)
            # --spatial-aug: the two spatial ops the affine above does not
            # cover, from the CSIRO-biomass 2nd place's ablation (their two
            # strongest augs). (1) A zoom-IN crop, area U(0.8, 1.0) at a
            # random centre, same window for every slice of the series - the
            # affine's scale tops out at 1.08 and never pushes anatomy off
            # the frame. (2) Per-SLICE occlusion boxes (RandomErasing): MRI
            # artefacts (motion, susceptibility, pulsation ghosts) occlude
            # single slices in exactly this shape, and erasing teaches the
            # slice-sequence model to use through-plane redundancy.
            if self.domain_rand > 0 and random.random() < self.domain_rand:
                v = _domain_rand_series(v)
            if self.spatial_aug > 0 and random.random() < self.spatial_aug:
                import cv2
                area = random.uniform(0.8, 1.0)
                side = max(int(round(self.size * np.sqrt(area))), 1)
                y0 = random.randrange(self.size - side + 1)
                x0 = random.randrange(self.size - side + 1)
                v = np.stack([cv2.resize(np.ascontiguousarray(
                                             sl[y0:y0 + side, x0:x0 + side]),
                                         (self.size, self.size),
                                         interpolation=cv2.INTER_LINEAR)
                              for sl in v])
                if self.native_ctx:
                    # Erase the 24-STACK slice, then window: box i lands on
                    # every (window, channel) whose stack neighbour is slice
                    # i, exactly as _stack_ctx propagates an erased slice to
                    # its neighbours' channels in the off path. Draw all
                    # boxes first, in the off path's RNG order.
                    v = np.ascontiguousarray(v)
                    vw = v.reshape(self.n_slices, self.slice_ctx, *v.shape[1:])
                    nbr = self._stack_nbr()
                    boxes = {}
                    for si in range(self.n_slices):
                        if random.random() < 0.5:
                            boxes[si] = self._erase_box()
                    for w in range(self.n_slices):
                        for c in range(self.slice_ctx):
                            bx = boxes.get(int(nbr[w, c]))
                            if bx is not None:
                                ey, eh, ex, ew = bx
                                vw[w, c, ey:ey + eh, ex:ex + ew] = 0.0
                else:
                    for si in range(v.shape[0]):
                        if random.random() < 0.5:
                            ey, eh, ex, ew = self._erase_box()
                            v[si, ey:ey + eh, ex:ex + ew] = 0.0
        if self.xfuse:
            # augmentation (intensity offset, noise) must not resurrect an
            # absent partner: absent rows are EXACTLY zero, which is what the
            # model reads as "unpaired" (KneeNet xfuse branch).
            v = np.array(v, dtype=np.float32)
            v[self.n_slices:][~_xvalid] = 0.0
        v = np.ascontiguousarray(v)
        if self.native_ctx:
            v = v.reshape(self.n_slices, self.slice_ctx, *v.shape[1:])
        if not want_mask:
            return v
        if sm is None:
            return (v, np.zeros((self.n_slices, AUX_SEG_SIZE, AUX_SEG_SIZE), np.uint8),
                    False)
        import cv2
        seg = np.stack([cv2.resize(np.ascontiguousarray(sl),
                                   (AUX_SEG_SIZE, AUX_SEG_SIZE),
                                   interpolation=cv2.INTER_NEAREST)
                        for sl in sm]).astype(np.uint8)
        return v, seg, True

    def _stratified_slice_indices(self, z):
        """Return sorted native indices from independent integer depth bins.

        The integer bins partition ``range(z)`` using
        ``floor(arange(n_slices + 1) * z / n_slices)``. One independent
        ``random.randrange(lo, hi)`` draw selects a native centre from each
        disjoint bin. This gives unique, ordered centres whenever z is greater
        than n_slices, while leaving z <= n_slices on the legacy path.

        This helper is called only from the opt-in training branch. It is kept
        separate from the legacy path so the default data and RNG stream are
        byte-identical.
        """
        z = int(z)
        if z <= self.n_slices:
            raise ValueError("stratified slice sampling requires z > n_slices")
        edges = np.floor(
            np.arange(self.n_slices + 1, dtype=np.float64) * z
            / self.n_slices).astype(np.int64)
        idx = np.empty(self.n_slices, dtype=np.int64)
        for b in range(self.n_slices):
            idx[b] = random.randrange(int(edges[b]), int(edges[b + 1]))
        return idx

    # ---- --native-ctx helpers (scratch/nativectx) ----
    def _vshape(self):
        """Per-series output shape: (n, H, W), or (n, k, H, W) under native_ctx."""
        if self.native_ctx:
            return (self.n_slices, self.slice_ctx, self.size, self.size)
        if self.xfuse:
            return (2 * self.n_slices, self.size, self.size)
        return (self.n_slices, self.size, self.size)

    def _xfuse_partner(self, study, series_uid, idx, hw):
        """--xfuse: (n_slices, H, W) partner rows on this series' cached grid,
        plus a (n_slices,) bool validity. Linear interpolation between the two
        bracketing partner cached slices (frac from scratch/xfuse/
        build_pairs.py), then the in-plane registration M (cached A -> cached
        B) applied with WARP_INVERSE_MAP. No partner -> zeros, all invalid."""
        n = len(idx)
        out = np.zeros((n,) + tuple(hw), np.float32)
        valid = np.zeros(n, dtype=bool)
        e = self.xfuse_pairs.get(f"{study}/{series_uid}")
        if e is None:
            return out, valid
        partner, frac, M = e
        try:
            pv = np.load(os.path.join(self.vol_root, study, partner + ".npy"),
                         mmap_mode="r")
        except Exception:
            return out, valid
        import cv2
        zb = pv.shape[0]
        fi = frac[np.asarray(idx, dtype=int)]
        for i, f in enumerate(fi):
            if f < 0:
                continue
            lo = int(np.floor(f))
            w = float(f - lo)
            lo = min(max(lo, 0), zb - 1)
            hi = min(lo + 1, zb - 1)
            sl = np.asarray(pv[lo]).astype(np.float32) / 255.0
            if w > 1e-3 and hi != lo:
                sl = (1.0 - w) * sl + w * (np.asarray(pv[hi]).astype(np.float32) / 255.0)
            out[i] = cv2.warpAffine(sl, M, (int(hw[1]), int(hw[0])),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            valid[i] = True
        return out, valid

    def _ctx_index(self, idx, z, step=1):
        """(n_slices, slice_ctx) int array of cached-volume indices.

        z > n_slices (linspace regime, stride > 1): channel c of window i is
        the NATIVE neighbour clip(idx[i] + c - k//2, 0, z-1). z <= n_slices
        (the stack is the whole volume, replicate-padded via repeated idx, or
        exactly the volume, possibly with rounded train jitter): neighbours are
        taken in STACK space, idx[clip(i + c - k//2, 0, n-1)], which is exactly
        what _stack_ctx's replicate pad produces - so every series with
        z <= n_slices is bit-identical to the non-native path in train AND
        eval (the PREREG's built-in placebo); only z > n_slices series change.
        """
        k = self.slice_ctx
        off = np.arange(k) - k // 2
        idx = np.asarray(idx, dtype=int)
        if self.ctx_mm > 0:
            # physgeo: always NATIVE neighbours, `step` slices apart.
            return np.clip(idx[:, None] + off[None, :] * int(step), 0, z - 1)
        if z > self.n_slices:
            return np.clip(idx[:, None] + off[None, :], 0, z - 1)
        return idx[self._stack_nbr()]

    def _zstep(self, study, series_uid):
        """physgeo: native-slice step that best matches ctx_mm for this series."""
        if self.ctx_mm <= 0:
            return 1
        sp = self.zspacing.get(f"{study}/{series_uid}")
        return max(1, int(round(self.ctx_mm / sp))) if sp else 1

    def _stack_nbr(self):
        """(n_slices, slice_ctx) stack-space neighbour index clip(i + c - k//2)."""
        off = np.arange(self.slice_ctx) - self.slice_ctx // 2
        return np.clip(np.arange(self.n_slices)[:, None] + off[None, :], 0, self.n_slices - 1)

    def _erase_box(self):
        """One RandomErasing box (ey, eh, ex, ew); same draws as the off path."""
        a = random.uniform(0.02, 0.15) * self.size * self.size
        ar = float(np.exp(random.uniform(np.log(0.3), np.log(3.3))))
        eh = min(max(int(round(np.sqrt(a * ar))), 1), self.size)
        ew = min(max(int(round(np.sqrt(a / ar))), 1), self.size)
        ey = random.randrange(self.size - eh + 1)
        ex = random.randrange(self.size - ew + 1)
        return ey, eh, ex, ew

    def _rev_stack(self, v):
        """Sagittal stack reversal. Under native_ctx the channel axis is
        reversed too, so 'previous/next' keep their meaning in the reversed
        stack (what _stack_ctx would have built from a reversed 24-stack)."""
        if not self.native_ctx:
            if self.xfuse:
                n = self.n_slices          # reverse each half, keep halves
                return np.concatenate([v[:n][::-1], v[n:][::-1]], 0)
            return v[::-1]
        w = v.reshape(self.n_slices, self.slice_ctx, *v.shape[1:])
        return w[::-1, ::-1].reshape(-1, *v.shape[1:])

    def _build(self, study, do_flip):
        """Assemble one study's (vols, stypes, seqs, mask). Split out of
        __getitem__ so cutmix can build a PARTNER study the same way."""
        picked = self._pick_series(study)
        vols = np.zeros((self.n_series,) + self._vshape(), np.float32)
        stypes = np.zeros((self.n_series,), np.int64)
        seqs = np.zeros((self.n_series,), np.int64)
        mask = np.zeros((self.n_series,), np.float32)
        seg = seg_valid = None
        if self.aux_seg_root:
            seg = np.zeros((self.n_series, self.n_slices, AUX_SEG_SIZE, AUX_SEG_SIZE),
                           np.uint8)
            seg_valid = np.zeros((self.n_series,), np.float32)
        for j, s in enumerate(picked):
            if self.aux_seg_root:
                mp = os.path.join(self.aux_seg_root, study, s[0] + "_msk.npy")
                if not os.path.exists(mp):
                    mp = None
                vols[j], seg[j], sv = self._load_series(
                    study, s[0], s[1], flip=do_flip, want_mask=True, mask_path=mp)
                seg_valid[j] = 1.0 if sv else 0.0
            else:
                vols[j] = self._load_series(study, s[0], s[1], flip=do_flip)
            stypes[j] = STYPE_ID[(s[1], s[2])]
            seqs[j] = self.seq.get(f"{study}/{s[0]}", 0)
            mask[j] = 1.0
        # Protocol-composition robustness (--series-dropout): with prob P
        # drop ONE present series (mask -> 0; attention gives masked series
        # exactly zero weight, audited). Trains the fusion to work from
        # incomplete studies - unseen sites may lack series types, and the
        # private set's protocol mix is not ours to assume. Only when >=3
        # series are present, so no study degrades below the corpus minimum.
        if (self.series_dropout > 0 and len(picked) >= 3
                and random.random() < self.series_dropout):
            # Never drop slot 0: StudyEncoder unconditionally un-masks slot 0
            # (model.py `pad[:, 0] = False`, NaN guard), so a dropped slot 0
            # would re-enter attention as a zero-volume token with the real
            # series' type. Flagged by notes/masking_and_aux_review_20260907.md;
            # the shipped serdrop arm (p=0.5) carried this artefact on ~1/len
            # of its drops. Default series_dropout=0 -> this branch never runs.
            k = random.randrange(1, len(picked))
            vols[k] = 0.0
            mask[k] = 0.0
            if seg_valid is not None:
                seg_valid[k] = 0.0
        if self.tfcrop_root:
            if seg is not None or self.xfuse or self.native_ctx:
                raise NotImplementedError("--tfcrop-root is not wired with aux-seg/xfuse/native-ctx")
            cv, ct, cq, cm = self._tfcrop_slots(study, do_flip)
            vols = np.concatenate([vols, cv], 0)
            stypes = np.concatenate([stypes, ct], 0)
            seqs = np.concatenate([seqs, cq], 0)
            mask = np.concatenate([mask, cm], 0)
        return vols, stypes, seqs, mask, seg, seg_valid

    def _tfcrop_slots(self, study, do_flip):
        """TFCROP_SLOTS extra series from <tfcrop_root>/<study>.npz (uint8
        (Z, h, w) crops built by scratch/tfcrop_20261011/build_tfcrop.py at
        native DICOM resolution). Upsampled to (size, size), then pushed through
        the SAME _load_series ops as a Coronal series (slice pick, side canon,
        flip, train augs). Under flip-swap the lateral/medial TYPE ids swap with
        the labels. Absent crop -> zero volume, mask 0."""
        import cv2
        n = len(TFCROP_KEYS)
        vols = np.zeros((n,) + self._vshape(), np.float32)
        st = np.array([TFCROP_TYPE[k] for k in TFCROP_KEYS], np.int64)
        sq = np.zeros((n,), np.int64)
        m = np.zeros((n,), np.float32)
        fp = os.path.join(self.tfcrop_root, study + ".npz")
        if not os.path.exists(fp):
            return vols, st, sq, m
        z = np.load(fp)
        for j, k in enumerate(TFCROP_KEYS):
            if k not in z.files:
                continue
            a = z[k]
            a = np.stack([cv2.resize(np.ascontiguousarray(sl), (self.size, self.size),
                                     interpolation=cv2.INTER_LINEAR) for sl in a])
            vols[j] = self._load_series(study, "tfcrop_" + k, "Coronal", flip=do_flip, vol_arr=a)
            m[j] = 1.0
            src = str(z["src_" + k[-2:]]) if ("src_" + k[-2:]) in z.files else ""
            sq[j] = self.seq.get(f"{study}/{src}", 0)
            if do_flip:
                st[j] = TFCROP_TYPE[TFCROP_SWAP[k]]
        return vols, st, sq, m

    def _cut_box(self, lam):
        """Standard CutMix box (Yun et al.), sized so its area is 1 - lam."""
        r = np.sqrt(1.0 - lam)
        ch, cw = int(self.size * r), int(self.size * r)
        cy, cx = random.randrange(self.size), random.randrange(self.size)
        y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, self.size)
        x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, self.size)
        return y0, y1, x0, x1

    def __getitem__(self, i):
        study = self.studies[i]
        # One decision for the whole study - see FLIP_SWAP.
        do_flip = self.flip_swap > 0 and random.random() < self.flip_swap
        vols, stypes, seqs, mask, seg, seg_valid = self._build(study, do_flip)

        # ---- CutMix ------------------------------------------------------
        # RSNA 2025's 5th place measured cutmix p=1.0 as the best augmentation
        # FOR AUC specifically, beating flip, rotate, brightness, contrast, blur
        # and mixup; their 6th place put cutmix/mixup at about +0.01 CV. Our
        # entire policy before this was rotate +/-10, scale 0.92-1.08, shift
        # +/-4% at p=0.7, which is mild by their standards.
        #
        # Done in the DATASET, not the train loop, because bs=1: there is no
        # second sample in a batch to mix with, so the partner is drawn here.
        # The SAME box is pasted into every series and every slice, so the mix
        # is one coherent anatomical region rather than 144 unrelated patches -
        # otherwise the label mix would not correspond to anything the model
        # sees. lam is the retained-area fraction and the labels mix by it.
        lam = 1.0
        if self.cutmix > 0 and len(self.studies) > 1 and random.random() < self.cutmix:
            k = random.randrange(len(self.studies) - 1)
            k = k + 1 if k >= i else k                       # any study but this one
            other = self.studies[k]
            pv, _ps, _pq, pm, _pseg, _psegv = self._build(other, do_flip)
            lam0 = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            y0, y1, x0, x1 = self._cut_box(lam0)
            if y1 > y0 and x1 > x0:
                # Only paste into slots the PARTNER actually filled; pasting a
                # padded-zero region would teach "blank rectangle" as evidence.
                if self.xfuse:
                    _xz = vols[:, self.n_slices:].reshape(
                        self.n_series, self.n_slices, -1).max(-1) == 0
                for j in range(self.n_series):
                    if mask[j] > 0 and pm[j] > 0:
                        vols[j][..., y0:y1, x0:x1] = pv[j][..., y0:y1, x0:x1]
                if self.xfuse:
                    vols[:, self.n_slices:][_xz] = 0.0
                # True retained fraction, from the clipped box - not lam0, which
                # is wrong whenever the box runs off the edge.
                lam = 1.0 - ((y1 - y0) * (x1 - x0)) / float(self.size * self.size)
                self._mix_uid = other
            else:
                self._mix_uid = None
        else:
            self._mix_uid = None

        out = {
            "vol": torch.from_numpy(vols),
            "stype": torch.from_numpy(stypes),
            "seq": torch.from_numpy(seqs),
            "mask": torch.from_numpy(mask),
            "study": study,
        }
        if self.aux_seg_root:
            out["seg"] = torch.from_numpy(seg)
            out["seg_valid"] = torch.from_numpy(seg_valid)
        if self.labels is not None:
            yv = self.labels[study].astype(np.float32).copy()
            if self.label_repair:
                yv, repair_w, repair_lo, repair_hi, repair_interval = \
                    self._label_repair_arrays(study, yv)
            # Mix the target by the retained-area fraction. Done BEFORE the
            # flip swap so both studies' labels are in the same (unflipped)
            # frame; the swap then applies to the mixture, which is correct
            # because both volumes were built with the same do_flip.
            if lam < 1.0 and self._mix_uid is not None and self._mix_uid in self.labels:
                if self.label_repair:
                    py, pw, plo, phi, pint = self._label_repair_arrays(
                        self._mix_uid,
                        self.labels[self._mix_uid].astype(np.float32))
                    yv = lam * yv + (1.0 - lam) * py
                    repair_w = lam * repair_w + (1.0 - lam) * pw
                    repair_lo = lam * repair_lo + (1.0 - lam) * plo
                    repair_hi = lam * repair_hi + (1.0 - lam) * phi
                    repair_interval = repair_interval | pint
                else:
                    yv = lam * yv + (1.0 - lam) * self.labels[self._mix_uid].astype(np.float32)
            fw = np.ones(len(FINDINGS), np.float32)
            if do_flip:
                for a, b_ in FLIP_SWAP:
                    yv[a], yv[b_] = yv[b_], yv[a]
                    if self.label_repair:
                        repair_w[a], repair_w[b_] = repair_w[b_], repair_w[a]
                        repair_lo[a], repair_lo[b_] = repair_lo[b_], repair_lo[a]
                        repair_hi[a], repair_hi[b_] = repair_hi[b_], repair_hi[a]
                        repair_interval[a], repair_interval[b_] = \
                            repair_interval[b_], repair_interval[a]
                for k in FLIP_MASK:
                    fw[k] = 0.0
            # Coverage multiplies rather than overwrites, so an external study
            # that is BOTH flipped and partially scored keeps both masks.
            if self.coverage is not None:
                fw = fw * self.coverage.get(study, _FW_ONES)
            if self.label_repair:
                fw = fw * repair_w
                out["lr_lo"] = torch.from_numpy(repair_lo)
                out["lr_hi"] = torch.from_numpy(repair_hi)
                out["lr_interval"] = torch.from_numpy(repair_interval)
            out["y"] = torch.from_numpy(yv)
            # Always emitted when flipping is enabled, so the collate shape is
            # constant and the train loop needs no per-batch branch.
            if self.flip_swap > 0 or self.coverage is not None or self.label_repair:
                out["fw"] = torch.from_numpy(fw)
        if self.agree:
            out["aw"] = torch.from_numpy(
                self.agree.get(study, np.ones(len(FINDINGS), np.float32)))
        if self.aux_labels:
            a = self.aux_labels.get(study)
            # have=0 marks a study with no auxiliary extraction so the loss can
            # exclude it; the target itself is a harmless zero vector.
            av = (a if a is not None
                  else np.zeros(len(next(iter(self.aux_labels.values()))), np.float32)).copy()
            if lam < 1.0 and self._mix_uid is not None:
                pa = self.aux_labels.get(self._mix_uid)
                if pa is not None and a is not None:
                    av = lam * av + (1.0 - lam) * pa.astype(np.float32)
            if do_flip:
                for x, yb in self.aux_flip_swap:
                    if max(x, yb) < len(av):
                        av[x], av[yb] = av[yb], av[x]
            out["y_aux"] = torch.from_numpy(av)
            out["has_aux"] = torch.tensor(1.0 if a is not None else 0.0)
        if self.pairmax:
            t = self.pairmax.get(study)
            # max-over-pair supervision for labels that name EITHER
            # meniscus without saying which (MRNet). Symmetric under the
            # medial/lateral swap, so flip_swap needs no special casing.
            out["y_pairmax"] = torch.tensor(t if t is not None else 0.0)
            out["has_pairmax"] = torch.tensor(1.0 if t is not None else 0.0)
        return out


def load_series_quality(path="data/processed/series_geom.json"):
    """{study/series: (thickness, -rows*cols, spacing)} = serve-time quality key
    for --pick-best-twin (scratch/twinserve). LOWER tuple = better: thinner
    slices first, then a larger acquisition matrix, then finer in-plane pixels.

    Source is the same serve-available header table --ctx-mm / --aspect-fix
    already read (data/processed/series_geom.json, 24,371/24,371 RSNA series
    complete on thickness/rows/cols/spacing). A series with ANY of the four
    missing is omitted; _pick_series then leaves that whole type group in its
    OLD order, so the flag is a strict no-op wherever the table is incomplete
    (e.g. external co-training pools that have their own geom file).
    """
    try:
        out = {}
        for r in json.load(open(path)):
            t, rw, cl, sp = (r.get("thickness"), r.get("rows"),
                             r.get("cols"), r.get("spacing"))
            if not t or not rw or not cl or not sp:
                continue
            out[f"{r['study']}/{r['series']}"] = (float(t), -float(rw) * float(cl),
                                                  float(sp))
        return out
    except Exception:
        return {}


def load_xfuse_pairs(path="scratch/xfuse/xfuse_pairs.json"):
    """--xfuse pair table (scratch/xfuse/build_pairs.py) -> {study/FS_series:
    (partner_uid, frac float64 (z_cached,), M float64 (2,3))}."""
    import json
    with open(path) as fh:
        d = json.load(fh)
    return {k: (e["partner"], np.asarray(e["frac"], np.float64),
                np.asarray(e["M"], np.float64)) for k, e in d.items()}


def load_series_index(series_csv):
    import csv
    idx = {}
    with open(series_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx.setdefault(r["StudyInstanceUID"], []).append(
                (r["SeriesInstanceUID"], r["Anatomical_Plane"],
                 int(r["Fluid_Sensitive"])))
    return idx


def load_evidence_sampling_map(path):
    """Load a verified native-to-cache support map for training-only sampling.

    The map is intentionally separate from label repairs.  It must be produced
    by the packet builder after native/cache correspondence was checked; a
    missing or unverified map is an error rather than a reason to guess.
    Keys are ``study_uid/series_uid`` and values are ``[[lo, hi], ...]`` in
    native cached-slice indices.  Inference never consumes this map.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("evidence sampling map must have schema_version=1")
    if payload.get("native_to_cache_verified") is not True:
        raise ValueError("evidence sampling map is not native-to-cache verified")
    intervals = payload.get("intervals")
    if not isinstance(intervals, dict) or not intervals:
        raise ValueError("evidence sampling map has no intervals")
    out = {}
    for key, values in intervals.items():
        if not isinstance(key, str) or not key or not isinstance(values, list) or not values:
            raise ValueError(f"invalid evidence interval entry {key!r}")
        clean = []
        for pair in values:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)
                    or pair[0] < 0 or pair[1] < pair[0]):
                raise ValueError(f"invalid evidence interval for {key!r}")
            clean.append([int(pair[0]), int(pair[1])])
        out[key] = clean
    return out


def evidence_slice_indices(z, n_slices, standard_idx, intervals):
    """Keep half of standard centres and replace half by support centres.

    ``standard_idx`` is the already-created standard train grid, so this helper
    does not alter the standard RNG path.  The returned centres are sorted to
    preserve native through-plane order.  Callers must provide a verified map.
    """
    z, n_slices = int(z), int(n_slices)
    idx = np.asarray(standard_idx, dtype=np.int64)
    if z <= n_slices or idx.size != n_slices:
        return idx
    if not intervals:
        raise ValueError("evidence sampling requires at least one support interval")
    lo = min(int(x[0]) for x in intervals)
    hi = max(int(x[1]) for x in intervals)
    if lo < 0 or hi < lo or hi >= z:
        raise ValueError(f"evidence interval [{lo}, {hi}] outside native depth {z}")
    keep_n = n_slices // 2
    keep = idx[:keep_n]
    replacement = np.linspace(lo, hi, n_slices - keep_n).round().astype(np.int64)
    return np.sort(np.concatenate([keep, replacement]))


def load_report_labels(path, prob_table, silent=None):
    """report_labels.jsonl -> {study: soft target vector}.

    Rows may carry EITHER "grades" (ordinal -1..3, mapped through the
    calibrated table - the original path) or "probs" (direct continuous
    targets in [0,1], the observer-v5 path, used verbatim). A file may mix
    both: competition rows with probs, external co-training rows with
    grades. Direct probs bypass the table AND the silent prior entirely -
    the observer model already encodes silence/denial semantics.

    A row may also carry BOTH, in which case grades set all twelve and
    "probs" OVERRIDES only the keys it names. That is what the quantitative
    X-ray refinement needs: the MOAKS ladder still supplies ten columns and
    the coverage mask, while the two compartment columns carry a continuous
    target that no ordinal grade can express. A probs-only row is unchanged -
    it still has to name all twelve.
    """
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "probs" in r and "grades" not in r:
                out[r["uid"]] = np.array(
                    [float(r["probs"][k]) for k in FINDINGS], dtype=np.float32)
            else:
                g = [r["grades"].get(k, -1) for k in FINDINGS]
                v = grade_to_prob(g, prob_table, silent)
                for k, pv in (r.get("probs") or {}).items():
                    v[FINDINGS.index(k)] = np.float32(pv)
                out[r["uid"]] = v
    return out


def fit_silent_table(pred_path, gold_path="data/processed/gold58.json",
                     pooled=None, k=10.0, exclude=()):
    """Per-finding P(gold=1 | report is silent), shrunk toward the pooled value.

    Why only the -1 cell. Pooling the whole grade ladder is right: 58 studies x
    12 findings supports five grade-level rates, not sixty. But the -1 cell is
    where the counts are LARGEST (Synovitis n=31, Baker's n=30, Lateral OA
    n=19), and where the pooled value is most wrong - measured, silence carries
    P(gold=1) of 0.258 for Synovitis and 0.250 for PF OA against 0.000 for ACL,
    Medial OA, Lateral Meniscus and Effusion. One number cannot serve both.

    Shrinkage is (n*p_hat + k*pooled)/(n + k), so a finding with two silent
    studies barely moves and one with thirty mostly gets its own rate.

    This reads gold labels, so it reintroduces the calibration leak that
    --calib fixed removed. It is opt-in for exactly that reason, and the
    leaderboard - not gold-58 - has to arbitrate whether it helps.
    """
    pooled = PRIOR_TABLE[-1] if pooled is None else pooled
    gold = {r["uid"]: r["labels"] for r in json.load(open(gold_path))}
    exclude = set(exclude)
    num = {f: 0.0 for f in FINDINGS}
    den = {f: 0.0 for f in FINDINGS}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["uid"] not in gold or r["uid"] in exclude:
                continue
            for f_ in FINDINGS:
                if int(r["grades"].get(f_, -1)) == -1:
                    num[f_] += gold[r["uid"]][f_]
                    den[f_] += 1
    table, counts = [], {}
    for f_ in FINDINGS:
        n = den[f_]
        table.append((num[f_] + k * pooled) / (n + k))
        counts[f_] = int(n)
    return np.array(table, np.float32), counts


def load_agreement(path="data/processed/label_agreement.npz", floor=0.3):
    """{study: np.array(12,)} per-cell label confidence from repeated extraction.

    Running the SAME prompt three times disagrees on 10.4% of study-finding
    cells - 26% on Fracture, 20% on Contusion. Those cells are coin flips, and
    training on them at full weight teaches noise.

    Weight is floor + (1-floor)*agreement, so an unanimous cell keeps weight 1
    and a fully-split one drops to `floor`. Studies absent from the file get 1.0:
    no information about a study is not evidence against it.

    Unlike a per-FINDING shrinkage this is per-CELL, so it changes the relative
    weight of studies WITHIN a finding. That distinction matters - a uniform
    per-finding rescale is a monotone transform and cannot change what the model
    ranks, which is why --mute-after and the ranking-loss gate both measured null.
    """
    try:
        d = np.load(path, allow_pickle=True)
        a = d["agree"].astype(np.float32)
        w = floor + (1.0 - floor) * a
        return {str(u): w[i] for i, u in enumerate(d["uids"])}
    except Exception:
        return {}


def load_aux_labels(path, table=None):
    """{study: np.array(n_aux,)} soft targets for the auxiliary report fields.

    Same grade ladder as the scored findings, so the two supervision signals
    share a calibration. Returns ({}, []) when the file is absent, which
    disables the head rather than failing - the auxiliary targets are extra
    supervision, never a requirement.
    """
    from extract_aux import AUX_FINDINGS
    table = PRIOR_TABLE if table is None else table
    out = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                g = r["grades"]
                # null grade -> NaN -> zero per-cell loss weight in train.py.
                # Distinct from -1 ("report silent"), which is a real 0.10
                # target; null means "this source never scores this field"
                # (MOAKS has no LCL/PLC), where a fabricated soft negative
                # would be the cotrain-val trap in aux form.
                out[r["uid"]] = np.array(
                    [np.nan if g.get(f, -1) is None else table[int(g.get(f, -1))]
                     for f in AUX_FINDINGS], np.float32)
    except Exception:
        return {}, []
    return out, AUX_FINDINGS


def load_gold(path="data/processed/gold58.json"):
    g = json.load(open(path))
    return {r["uid"]: np.array([r["labels"][k] for k in FINDINGS], np.float32)
            for r in g}


# Leak-free grade -> probability ladder. Hand-specified and deliberately NOT
# fitted: the gold studies are our only honest evaluation set, and a table
# fitted on them flows into every training target and quietly inflates the gold
# AUC we then read. Measured against the fitted table on the real corpus
# (4,406 studies x 12 findings) this differs by a mean of 0.0088 and a max of
# 0.0225, with identical ordering - so the fitted version buys nothing that a
# rounded prior does not already give, and it costs the evaluation's integrity.
PRIOR_TABLE = {-1: 0.10, 0: 0.05, 1: 0.35, 2: 0.70, 3: 0.95}


def fit_prob_table(pred_path, gold_path="data/processed/gold58.json",
                   mode="fixed", exclude=()):
    """P(gold=1 | report grade), pooled over findings.

    mode="fixed"  - PRIOR_TABLE, touches no gold label. Default, and the only
                    mode under which the gold-58 AUC means what it says.
    mode="gold"   - fitted on all 58. Leaks the evaluation set into the training
                    targets; kept only so earlier runs stay reproducible.
    mode="kfold"  - fitted on the gold studies NOT in `exclude`. For a genuine
                    cross-fit: train one model per fold and score each only on
                    its own held-out gold studies.

    A note on why there is no leave-one-out mode. LOO over this table cannot fix
    the leak on its own, because the table is consumed by the 4,349 *training*
    studies and the model then predicts all 58. Removing study i from the table
    does nothing unless a model is retrained without it, so honest LOO means 58
    training runs. K-fold with K models is the affordable form of the same idea.

    Pooling over findings is deliberate: 58 x 12 is 696 observations, which
    supports five grade-level rates but not five per finding. A Laplace prior
    keeps grades that are rare in the gold set from collapsing to 0 or 1.
    """
    if mode == "fixed":
        return dict(PRIOR_TABLE), {}
    if mode not in ("gold", "kfold"):
        raise ValueError(f"unknown calibration mode {mode!r}")

    gold = json.load(open(gold_path))
    gold = {r["uid"]: r["labels"] for r in gold}
    exclude = set(exclude)
    num, den = {}, {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["uid"] not in gold or r["uid"] in exclude:
                continue
            for k in FINDINGS:
                g = int(r["grades"].get(k, -1))
                num[g] = num.get(g, 0) + gold[r["uid"]][k]
                den[g] = den.get(g, 0) + 1
    if len(den) < 5:
        raise ValueError(
            f"calibration saw only grades {sorted(den)} - the labels file does "
            f"not overlap the gold studies, so every target would collapse to "
            f"the 0.05 fallback and training would run on a signal-free set")
    table = {g: (num[g] + 1.0) / (den[g] + 2.0) for g in den}
    # Any grade absent from the fitting split still needs a value.
    for g, v in PRIOR_TABLE.items():
        table.setdefault(g, v)
    return table, {g: den[g] for g in den}


def gold_calib_split(gold_uids, fold, n_folds):
    """Deterministic gold split for mode="kfold": (fit_on, held_out)."""
    us = sorted(gold_uids)
    held = {u for i, u in enumerate(us) if i % n_folds == fold}
    return [u for u in us if u not in held], sorted(held)
