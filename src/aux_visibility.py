"""Visibility-conditioned aux supervision (scratch/auxvis, 2026-09-13).

The per-series aux head predicts all twelve study labels for EVERY series
(train.py: ``y.unsqueeze(1).expand_as(aux)``). A study label broadcast to a
series whose plane / sequence cannot show the finding is label noise on the
aux head, which shares the trunk. VIS[stype, finding] = 1 where the finding
is anatomically assessable on that series type, 0 otherwise; ``--aux-
visibility-mask`` multiplies the aux BCE by it and drops the masked cells
from the mean's denominator.

Series type = data.STYPE_ID[(plane, fluid_sensitive)] = plane*2 + fluid with
plane in (Sagittal, Coronal, Axial) and fluid in (0 = T1/PD, 1 = T2-FS /
PD-FS / STIR). Finding order = data.FINDINGS.

Table authored 2026-09-13 from radiology convention (the 2026-09-07 review
note notes/masking_and_aux_review_20260907.md proposed visibility-conditioned
aux supervision but drafted no per-plane table):
  ACL              sagittal, either sequence
  MCL              coronal, either sequence
  Med/Lat meniscus sagittal + coronal
  Med/Lat OA       coronal + sagittal (cartilage / osteophytes)
  PF OA            axial + sagittal
  Effusion         every series (fluid-weighted stronger, but kept everywhere)
  Synovitis        fluid-sensitive sequence, any plane
  Baker's          sagittal + axial
  Contusion        fluid-sensitive sequence (marrow oedema), any plane
  Fracture         every series (cortical break visible on all)
"""
import numpy as np

from data import FINDINGS, PLANES, SERIES_TYPES

_SAG, _COR, _AX = PLANES
# finding -> set of (plane, fluid) cells that are visible
_RULES = {
    "ACL": {(_SAG, 0), (_SAG, 1)},
    "MCL": {(_COR, 0), (_COR, 1)},
    "Medial Meniscus": {(_SAG, 0), (_SAG, 1), (_COR, 0), (_COR, 1)},
    "Lateral Meniscus": {(_SAG, 0), (_SAG, 1), (_COR, 0), (_COR, 1)},
    "Medial OA": {(_SAG, 0), (_SAG, 1), (_COR, 0), (_COR, 1)},
    "Lateral OA": {(_SAG, 0), (_SAG, 1), (_COR, 0), (_COR, 1)},
    "PF OA": {(_SAG, 0), (_SAG, 1), (_AX, 0), (_AX, 1)},
    "Effusion": set(SERIES_TYPES),
    "Synovitis": {(p, 1) for p in PLANES},
    "Baker's": {(_SAG, 0), (_SAG, 1), (_AX, 0), (_AX, 1)},
    "Contusion": {(p, 1) for p in PLANES},
    "Fracture": set(SERIES_TYPES),
}
assert set(_RULES) == set(FINDINGS), "visibility table must cover data.FINDINGS exactly"


def visibility_table():
    """(n_series_types=6, n_findings=12) float32 array, 1 = visible."""
    t = np.zeros((len(SERIES_TYPES), len(FINDINGS)), np.float32)
    for j, f in enumerate(FINDINGS):
        for i, st in enumerate(SERIES_TYPES):
            if st in _RULES[f]:
                t[i, j] = 1.0
    return t


def describe():
    t = visibility_table()
    rows = ["[aux-vis] table (rows = series type plane/fluid, cols = "
            + ",".join(f[:4] for f in FINDINGS) + ")"]
    for i, (p, fl) in enumerate(SERIES_TYPES):
        rows.append(f"[aux-vis]   {p[:3]}/{'fluid' if fl else 'nonfl'}: "
                    + " ".join(str(int(v)) for v in t[i]))
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
