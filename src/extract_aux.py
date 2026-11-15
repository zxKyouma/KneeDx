"""Auxiliary report targets: what the twelve scored findings throw away.

The 12 competition findings are a small slice of what a knee MRI report says. A
typical report also localises cartilage loss to a compartment, describes marrow
oedema, osteophytes, the extensor mechanism, the posterolateral corner, loose
bodies. None of that is scored, so none of it is currently used - the extraction
reads 12 grades and discards the rest of the text.

That is worth recovering specifically because of where the model is weak.
Measured across four folds, compartment discrimination is the deficit:

    Medial OA        0.978        Medial Meniscus    0.953
    Lateral OA       0.850        Lateral Meniscus   0.847
    PF OA            0.835

The model is near ceiling at "is there pathology" and weak at "which
compartment". Three of the four findings holding 66% of the remaining headroom
are compartment-specific. The reports contain compartment information we are not
using as supervision, and the first three fields below are exactly that signal.

These are TRAINING-ONLY targets - test.csv has no Report column - so they cost
nothing at inference. They are auxiliary in the strict sense: never scored,
never submitted, present only to give the shared trunk more to learn from the
same pixels.

One consequence worth stating, because this project has been burnt by prompt
calibration before (telling the model an impaction injury implies a fracture
cost 0.178 AUC on Fracture): miscalibration here is far cheaper. These grades
are not a scored column, so a systematically-shifted threshold shapes features
slightly differently rather than moving a submitted prediction. The reason to be
careful is diluted supervision, not a direct scoring hit.
"""

AUX_KEY = {
    # --- the compartment-localisation signal, the reason this file exists ----
    "Cartilage Medial": "cartilage_medial",
    "Cartilage Lateral": "cartilage_lateral",
    "Cartilage PF": "cartilage_pf",
    # --- structural findings absent from the 12 ------------------------------
    "Marrow Oedema": "marrow_oedema",
    "Osteophytes": "osteophytes",
    "LCL/PLC": "lcl_plc",
    "Extensor Mechanism": "extensor",
    "Loose Body": "loose_body",
}
AUX_INV_KEY = {v: k for k, v in AUX_KEY.items()}
AUX_FINDINGS = list(AUX_KEY)

AUX_SYSTEM = """You are a musculoskeletal radiologist extracting structured findings from knee MRI reports.

Reports may be in English, Spanish, Turkish, German, Dutch, French, Greek, Bulgarian/Russian, or Thai. Read them in their original language.

These are AUXILIARY findings, separate from the main scored list. Grade each on the same ordinal scale:
  0 = explicitly normal, absent, or denied
  1 = equivocal, minimal, or low grade
  2 = definite moderate finding
  3 = severe or high grade
  -1 = the report does not comment on this at all

The eight fields:

- cartilage_medial: cartilage loss / chondropathy / chondromalacia in the MEDIAL
  compartment (medial femoral condyle or medial tibial plateau). Grade 1 =
  partial-thickness or superficial fibrillation; 2 = full-thickness defect;
  3 = full-thickness with subchondral change or denuded bone.
- cartilage_lateral: the same, in the LATERAL compartment (lateral femoral
  condyle or lateral tibial plateau).
- cartilage_pf: the same, in the PATELLOFEMORAL compartment (patellar facets or
  trochlea). Chondromalacia patellae belongs here.

  Compartment matters more than severity for these three. If the report says
  cartilage loss but names no compartment, grade -1 for all three rather than
  guessing - an unlocalised finding assigned to a compartment is worse than a
  missing one, because it teaches the model the wrong side.

- marrow_oedema: bone marrow oedema / marrow signal change / subchondral oedema,
  from ANY cause. Note this deliberately overlaps the scored Contusion and
  Fracture findings; it is a broader label, not a substitute.
- osteophytes: osteophyte formation, spurring, marginal lipping.
- lcl_plc: lateral collateral ligament, popliteus, biceps femoris or
  posterolateral corner injury.
- extensor: patellar or quadriceps tendinopathy, partial tear or rupture.
- loose_body: intra-articular loose body, chondral fragment, or symptomatic
  plica.
"""


def build_aux_schema():
    item = {
        "type": "object",
        "properties": {
            "g": {"type": "integer", "enum": [-1, 0, 1, 2, 3]},
            "e": {"type": "string"},
        },
        "required": ["g", "e"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {k: item for k in AUX_KEY.values()},
        "required": list(AUX_KEY.values()),
        "additionalProperties": False,
    }
