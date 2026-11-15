"""One checkpoint -> model construction path, shared by every consumer.

Why this file exists. `src/export_oof.py` and `src/eval_checkpoints.py` each
grew their own copy of "rebuild the trained model from a .pt", and the copies
drifted: the OOF exporter read `gem_pool`/`routing_prior`/`slice_ctx`/
`band_pool`/`report_aux` off the checkpoint while the gold-58 scorer did not,
and the gold scorer read `n_out`/`moaks_blocks` while the exporter did not.
Two failure shapes followed (scratch/measaudit/GOLDSKIP_AUDIT_20260914.md):

  LOUD-then-silenced - a key the default constructor never created made
  `strict=True` raise, and eval_checkpoints turned that into
  `print("SKIPPING ...")` + `continue`. The member vanished from the npz and
  the package assay then compared two rosters that were both short. unionpack
  (margin 0.00011) and studblend were decided on packages missing four of
  their own new members; raux, a SHIPPING member, is absent from 0/38 gold npz.

  SILENT - a PARAMETER-FREE flag (`gem_pool`, `routing_prior`, `osseous_factor`,
  `oa_gate`, `stack_depth`, `slice3d="shift"`, `band_delta`, `study_pool`)
  cannot be sniffed from a state dict at all. Omitting it rebuilds a different
  model that strict-loads cleanly and prints a plausible, wrong number.

So: every model-affecting flag is resolved HERE, from `ck["args"]` with the
train.py default as the fallback, and a load failure raises. There is no skip
path. A missing/unexpected key means the reconstruction is wrong, and a wrong
reconstruction must cost a job, not a verdict.

Two flags stay sniffed from the weights rather than args, deliberately: the
slice encoder (`series_enc.rnn.*` / `series_enc.tr.*`) and the slice-position
table shape, because both are fully recoverable from the tensors and some early
checkpoints predate the args that name them.
"""

import os

import torch
import torch.nn as nn

from model import KneeNet

# src/train.py argparse defaults for every model-affecting flag, plus the two
# KneeNet-only ones (band_delta, n_out). A value equal to its default is not
# printed; anything else is, so the log says what was actually built.
TRAIN_DEFAULTS = {
    "slice_ctx": 3,
    # --img-norm (ev2sfix): non-persistent input-normalisation buffer, NO weight signature -> args is the only
    # contract. Kaggle kernel: src/kernel_imgnorm.py (ported 2026-10-14); unported stages refuse (kernel_refuse_imgnorm).
    "img_norm": "imagenet",
    "stack_depth": False,
    # --bn-valid-only (scratch/truebatch/PREREG.md): the backbone runs only on
    # mask!=0 series; padded rows get zero features. No weight signature, so
    # args is the ONLY contract. train.py SUPPRESSes the key when off, so old
    # checkpoints lack it and resolve to False here.
    "bn_valid_only": False,
    # --seq-adapt R (scratch/seqadapt/PREREG.md): per-series-type FiLM + rank-R
    # adapters hooked onto backbone stages. SUPPRESSed when off -> 0 here.
    # Cross-check: seq_adapt_mods.* weights (refuse on disagreement).
    "seq_adapt": 0,
    # --xfuse (scratch/xfuse/PREREG.md): 2*slice_ctx-channel stem + a
    # zero-init xfuse_emb, AND a data-path change (the loader stacks each FS
    # series with its geometry-matched non-FS partner; consumers must build
    # the dataset with xfuse_pairs=load_xfuse_pairs(ck args xfuse_pairs)).
    # Args are the contract; xfuse_emb.weight is the cross-check (kw_xfuse).
    "xfuse": False,
    "study_pool": "route",
    "band_pool": False,
    "band_delta": 0.125,
    "routing_prior": 0.0,
    "flat_attn": False,
    "cell_attn": 0,
    "cell_stage": -1,
    "gate_init": 0.1,
    "cell_slice_bands": 3,
    "gem_pool": 1.0,
    "osseous_factor": False,
    "oa_gate": False,
    "query_family": "none",
    "slice3d": "none",
    "blurpool": 0,
    "qsa": "none",
    "qsa_gate_init": 0.1,
    "aux_seg_w": 0.0,
    "drop_path": 0.0,
    # --evidence-mil (scratch/evmil): replaces the study transformer with a
    # per-pathology evidence head. study_enc and aux_head become None, so a
    # checkpoint built without it has an entirely different key set - the loud
    # signature. The weights do carry ev_R/ev_head.*, but args is the contract.
    "evidence_mil": False,
    # --bypass-study-transformer (scratch/evmil_bypass): attribution control
    # for evidence_mil. It DELETES study_enc.tr.* and nothing else, so unlike
    # gem_pool/osseous_factor it does have a weight signature; args is still
    # the contract and a disagreement with the weights is REFUSED, never
    # guessed (both directions checked below).
    "bypass_study_transformer": False,
    # --evidence-mil-v2 (scratch/evmil_v2): off|evw|mean. The two ON modes have
    # IDENTICAL parameter sets (they differ only in how alpha is computed in the
    # forward), so this is the gem_pool/osseous_factor hazard class squared: the
    # weights prove that SOME mode is on but can never say which. args is the
    # only contract, and a disagreement is refused, never guessed.
    "evidence_mil_v2": "off",
    # --evmil-prior (scratch/fracprior, scratch/priorpartner): which clinical
    # reliability prior INITIALISED ev_R. ev_R is a saved nn.Parameter, so a
    # rebuild loads the TRAINED values and this flag cannot change any score -
    # it is the record of which prior ran, and it is passed through so the
    # constructor's validation still refuses an unknown name rather than
    # silently falling back to "clinical".
    "evmil_prior": "clinical",
    # --evmil-tau-init / --evmil-dif-init (scratch/evpool): the init of the
    # focal LSE temperature ev_tau and of the diffuse mixture logits ev_mix.
    # Both are saved nn.Parameters, so a rebuild loads the TRAINED values and
    # neither flag can change any score - they are the record of which
    # aggregator flavour ran, passed through so the constructor's validation
    # refuses an unknown dif-init name rather than silently falling back to
    # "mean". Defaults reproduce the pre-flag init exactly.
    "evmil_tau_init": 1.0,
    "evmil_dif_init": "mean",
    # gamma init of the V2 residual. Affects TRAINING only (a rebuild loads the
    # trained gamma), but it is in args so the record says which init a run used.
    "ev2_gamma_init": 0.01,
    # --syno-specialist (scratch/synov2 Amendment 1): bundled Synovitis residual
    # (syno_* weights). It DOES have a weight signature, so both directions are
    # cross-checked and a disagreement is REFUSED, never guessed.
    "syno_specialist": False,
    "syno_gamma_init": 0.01,
    # --pfrel / --pfrel-generic (scratch/pfrel/PREREG.md). The mode is derived
    # in train.py from two mutually exclusive flags and stored as ONE string,
    # so a checkpoint can never claim both arms. The "rel" and "generic" key
    # sets differ (queries + cross-attention vs gpool + gmlp), which makes this
    # the rare parameter-BEARING flag - but args stays the contract and a
    # disagreement with the weights is REFUSED, never guessed (the
    # bypass_study_transformer precedent, both directions checked below).
    "pfrel": "off",
    "pfrel_lambda_pos": 0.1,
    "pfrel_beta": 1.0,
    # --bonewsrpn / --bone-dense (scratch/bonewsrpn/PREREG.md). Same pattern as
    # pfrel: two mutually exclusive flags collapse to ONE mode string in
    # train.py, args is the contract, the weights are the cross-check and a
    # disagreement is REFUSED, never guessed.
    "bone": "off",
    # --bone-series-fluid (scratch/bonefam): DATA-path only (state dict
    # identical); consumers pass it to KneeStudyDataset. Kernel refuses nonfs.
    "bone_series_fluid": "fs",
    # --tfcrop-root (scratch/tfcrop_20261011): DATA path (4 appended crop series,
    # consumers set KneeStudyDataset.TFCROP_ROOT) + MODEL path (ev_R gains 4
    # type columns). args is the contract, ev_R's width the cross-check.
    # Kernel NOT ported: its NOT_IMPLEMENTED table must refuse tfcrop_root.
    "tfcrop_root": "",
    "bone_k": 4,
    "bone_tau": 0.1,
    "bone_shell": 2.0,
    # scratch/divarch --ps25 (per-series late fusion). args is the contract; the
    # ps25_head.* weights are the cross-check (refuse on disagreement).
    "ps25": False,
    # --rubric-pool (scratch/rubpool/PREREG.md). Parameter-BEARING (rub_head.*,
    # rub_b) AND study_enc-deleting, so the weights carry an unambiguous
    # signature - but args stays the contract and a disagreement in EITHER
    # direction is REFUSED, never guessed (the evidence_mil precedent).
    "rubric_pool": False,
    # --ev-adapter (scratch/evform/PREREG.md): conditional evidence FORMATION,
    # a residual soft-routed low-rank adapter immediately BEFORE the per-slice
    # evidence readout. PARAMETER-BEARING (ev_adapter_mod.* keys) and on the
    # SERVE path, so - the pfrel/bone precedent - args is the contract, the
    # weights are the cross-check and a disagreement in either direction is
    # REFUSED, never guessed. "mlp" and "moe" do NOT share a key set (the moe
    # alone has ev_adapter_mod.route.*), so both are cross-checked.
    "ev_adapter": "off",
    "ev_adapter_k": 4,
    "ev_adapter_width": 32,
    # --darc-local (scratch/shortcutarms/PREREG.md §3, HELD): SERVE-PATH
    # evidence transform with NO weight signature (zero parameters, the keep
    # mask is a non-persistent buffer), so args is the ONLY contract and it
    # must be resolved here - an unlisted key would silently build a plain EV
    # member. The Kaggle kernel does not implement it: its NOT_IMPLEMENTED
    # table must refuse darc_local (or the checkpoint must be exported with
    # the rows folded into ev_head.weight).
    "darc_local": False,
}

# KNOWN TRAIN-ONLY flags: they change the OBJECTIVE, never the module graph,
# so the rebuilt model is identical with or without them and a checkpoint
# carrying one must build, not be refused. They are listed (rather than merely
# absent) so that "this key is not in TRAIN_DEFAULTS" is a decision on record
# instead of an oversight, and so the build log says which training objective
# produced the weights.
#   aux_visibility_mask - scratch/auxvis: 0/1 reweighting of the per-series
#     aux BCE cells.
#   auxbag - scratch/auxbag/PREREG.md: the per-series aux BCE is replaced by
#     ONE bag likelihood per finding, Q_c = 1 - prod_s (1 - q_sc). The aux
#     head/per-series evidence is not consumed at serve time, the bias init is
#     study-logit compensated, and no kernel port is required.
#   evdrop / evdrop_p - scratch/evdrop/PREREG.md: dominant-evidence dropout.
#     On a fraction evdrop_p of TRAINING forwards a second per-series pool is
#     built from the same slice tokens with the top-evidence slice of each
#     (study, series, focal finding) removed, and a focal-only study loss on
#     it is added. `self.training` gates the whole branch, no parameter or
#     buffer is created and the state dict is unchanged, so the serve forward
#     is byte-identical and no kernel port is required.
TRAIN_ONLY_FLAGS = {
    "aux_visibility_mask": False,
    "auxbag": False,
    "evdrop": False,
    "evdrop_p": 0.5,
    # scratch/witness/PREREG.md: the witness KL is a TRAINING loss term read
    # off out[1]. It adds no parameter, no buffer and no state-dict key, and
    # the serve/eval forward is byte-identical to an ordinary EV member, so a
    # checkpoint carrying these keys BUILDS (and the build log prints them)
    # rather than being refused, and no kernel port is required.
    "witness_targets": "",
    "witness_beta": 0.5,
    "witness_tau": 1.0,
    "witness_gate": "entropy",
    # scratch/slicewit/PREREG.md: the slice-witness KL is a TRAINING loss term
    # read off a forward hook on ev_head (output unmodified). No parameter, no
    # buffer, no state-dict key; the serve forward is byte-identical.
    "slicewit_targets": "",
    "slicewit_beta": 1.0,
    "slicewit_tau": 1.0,
    # scratch/shortcutarms/PREREG.md §PAT (src/pat.py): patching-based
    # counterfactual TRAINING. The patch view exists only in training forwards
    # (self.training gate), adds no parameter, buffer or state-dict key, and the
    # eval/serve forward is byte-identical (unit-tested), so no kernel port.
    "pat": False,
    "pat_grid": 2,
    "pat_w": 1.0,
    # scratch/qwenvit/PREREG.md: per-group grad clipping (backbone vs rest).
    # Optimiser-side only; no parameter, buffer or state-dict key.
    "clip_groups": False,
    # scratch/noisystud/PREREG_dose.md (Addendum D): step-matched auxiliary
    # pool stream. Data/loss-side only (an extra teacher-target forward per
    # main micro-batch); no parameter, buffer or state-dict key, serve forward
    # unchanged, no kernel port. main_order_hash is a log-only diagnostic.
    "aux_pool_teacher": "",
    "aux_pool_vol_root": "",
    "aux_pool_series_csv": "",
    "aux_pool_laterality_map": "",
    "aux_pool_every": 1,
    "aux_pool_w": 0.0,
    "main_order_hash": 0,
    # scratch/truebatch/PREREG.md: warmup+cosine LR schedule. Optimiser-side
    # only; no parameter, buffer or state-dict key.
    "lr_sched": "onecycle",
    # scratch/brierfork/PREREG.md: full-state fork/resume and the report-row
    # loss form (bce / 2(p-y)^2 / scaled BCE). Objective/optimiser-side only;
    # no parameter, buffer or state-dict key, serve forward unchanged.
    "fork_save_epoch": None,
    "fork_resume": "",
    "report_loss": "bce",
    "report_loss_scale": 1.0,
    "fork_measure_scale": 0,
    # scratch/cfconsist/PREREG.md: counterfactual half-frame-mask consistency
    # loss. A second TRAINING forward on a masked copy of the input plus a
    # penalty on the target-logit change; no parameter, buffer or state-dict
    # key, serve forward unchanged, no kernel port.
    # scratch/decouple_20261009/PREREG.md (CONDRES-R+): a TRAINING loss term on
    # the study logits; no parameter, buffer or state-dict key, serve forward
    # unchanged, no kernel port.
    "condres_w": 0.0,
    "condres_pairs": "",
    "condres_thr": 0.15,
    "condres_bank": 256,
    # scratch/latrank_20261011/PREREG.md (LATRANK): training loss term only.
    "latrank_w": 0.0,
    "latrank_bank": 64,
    "latrank_cap": 2.0,
    # scratch/rcwrlr_20261011/PREREG.md (RCWR-LR): training sampler only.
    "rcwr_resample": 0.0,
    # scratch/rcwrlr_20261011/AMENDMENT_1.md: target-set selector, sampler only.
    "rcwr_set": "v1",
    # scratch/srchead_20261011/PREREG.md (SRCHEAD): a SOURCE-SPECIFIC copy of
    # ev_head[2]'s OA+meniscus rows (srchead_lin.*) trained on OAI rows only.
    # The serve forward is the ordinary EV-MIL forward on ev_head (RSNA head);
    # build() pops srchead_lin.* into info["srchead_oai"] (refuse-not-skip: the
    # flag and the keys must agree), so the built model and the kernel are
    # unchanged apart from dropping two state-dict keys.
    "srchead": "",
    "cf_consist": 0.0,
    "cf_mask": "medial",
    "cf_target": "LatOA",
    "cf_mode": "drop",
    "cf_converse": False,
    "cf_prob": 1.0,
}
_dup = sorted(set(TRAIN_ONLY_FLAGS) & set(TRAIN_DEFAULTS))
if _dup:
    raise RuntimeError(f"flags declared both model-affecting and train-only: "
                       f"{_dup}")


def kw_rubpool(r, sd):
    a = bool(r.get("rubric_pool"))
    w = any(k.startswith("rub_head.") or k == "rub_b" for k in sd)
    if a and not w:
        raise ValueError("ck['args']['rubric_pool'] is set but the state_dict "
                         "carries no rub_head.*/rub_b weights; refusing to "
                         "build a guessed model")
    if w and not a:
        raise ValueError("state_dict carries rub_head.*/rub_b (rubric-pool "
                         "head) but ck['args']['rubric_pool'] is off/absent; "
                         "refusing to build a guessed model")
    if a and (bool(r.get("evidence_mil")) or ("ev_R" in sd)):
        raise ValueError("checkpoint claims BOTH rubric_pool and evidence_mil; "
                         "these are two different aggregators and the build "
                         "cannot choose")
    return a


def kw_evmil(r, sd):
    return bool(r.get("evidence_mil")) or ("ev_R" in sd)


def kw_tfcrop(r, sd):
    """--tfcrop-root: args is the contract, ev_R width (6 + 4) the cross-check;
    refuse-not-skip in both directions."""
    want = 4 if str(r["tfcrop_root"] or "") else 0
    have = (sd["ev_R"].shape[1] - 6) if "ev_R" in sd else 0
    if want != have:
        raise ValueError(f"tfcrop_root={r['tfcrop_root']!r} implies {want} crop types "
                         f"but ev_R has {have} extra columns - refusing")
    return want


def kw_evmil_v2(r, sd):
    m = str(r.get("evidence_mil_v2") or "off")
    if m not in ("off", "evw", "mean"):
        raise ValueError(f"ck['args']['evidence_mil_v2']={m!r} is not one of "
                         f"off|evw|mean; refusing to build a guessed model")
    has = any(k.startswith("ev2_") for k in sd)
    if has and m == "off":
        raise ValueError(
            "checkpoint carries ev2_* weights (evmil V2 morphology residual) "
            "but ck['args']['evidence_mil_v2'] is off/absent. evw and mean are "
            "structurally identical, so the mode cannot be recovered from the "
            "state_dict - refusing rather than serving the wrong aggregator.")
    if m != "off" and not has:
        raise ValueError(
            f"ck['args']['evidence_mil_v2']={m} but the state_dict has no ev2_* "
            f"weights; this checkpoint was not trained with the V2 residual.")
    return m


def kw_evadapter(r, sd):
    """--ev-adapter (scratch/evform/PREREG.md). args is the contract, the
    ev_adapter_mod.* weights are the cross-check; refuse-not-skip in BOTH
    directions, and refuse a mode/weight disagreement (only moe has a
    router)."""
    m = str(r.get("ev_adapter") or "off")
    if m not in ("off", "mlp", "moe"):
        raise ValueError(f"ck['args']['ev_adapter']={m!r} is not one of "
                         f"off|mlp|moe; refusing to build a guessed model")
    has = any(k.startswith("ev_adapter_mod.") for k in sd)
    if has and m == "off":
        raise ValueError(
            "checkpoint carries ev_adapter_mod.* weights (the evform "
            "conditional evidence-formation adapter) but ck['args']"
            "['ev_adapter'] is off/absent - refusing to serve a model the "
            "args do not describe.")
    if m != "off" and not has:
        raise ValueError(
            f"ck['args']['ev_adapter']={m!r} but the state_dict has no "
            f"ev_adapter_mod.* weights; this checkpoint was not trained with "
            f"the evform adapter.")
    if has and (m == "moe") != any(k.startswith("ev_adapter_mod.route.")
                                   for k in sd):
        raise ValueError(
            f"ev_adapter mismatch: ck['args'] says {m!r} but the weights "
            f"{'carry' if m != 'moe' else 'lack'} ev_adapter_mod.route.*")
    return m


def kw_synospec(r, sd):
    """args is the contract, syno_* weights are the cross-check; refuse both
    unrecoverable disagreements (src/ckpt_build.py refuse-not-skip)."""
    on = bool(r.get("syno_specialist"))
    has = any(k.startswith("syno_") for k in sd)
    if has and not on:
        raise ValueError(
            "checkpoint carries syno_* weights (the bundled Synovitis "
            "specialist residual) but ck['args']['syno_specialist'] is "
            "False/absent - refusing to serve a model the args do not describe.")
    if on and not has:
        raise ValueError(
            "ck['args']['syno_specialist'] is True but the state_dict has no "
            "syno_* weights; this checkpoint was not trained with the "
            "Synovitis specialist head.")
    return on


def kw_xfuse(r, sd):
    """args is the contract, xfuse_emb.weight the cross-check; any
    disagreement refuses (refuse-not-skip)."""
    on, has = bool(r["xfuse"]), "xfuse_emb.weight" in sd
    if on != has:
        raise ValueError(f"xfuse: ck args say {on} but xfuse_emb.weight "
                         f"{'present' if has else 'absent'} - refusing to guess")
    return on


def _getter(a):
    """ck['args'] is a dict in every current checkpoint but a Namespace in old ones."""
    if isinstance(a, dict):
        return lambda k, d=None: a.get(k, d)
    return lambda k, d=None: getattr(a, k, d)


def resolve_flags(sd, a):
    """Every KneeNet kwarg this checkpoint needs: args first, weights second.

    Returns (kwargs, nondefault) where nondefault is the subset of
    TRAIN_DEFAULTS that this checkpoint overrides, for logging.
    """
    g = _getter(a)
    if "study_enc.pool.w.weight" in sd:
        raise ValueError("legacy pooled checkpoint; score it with the "
                         "notebook's KneeNet, which keeps that path")

    # --- sniffed from the weights (recoverable, and older than their args) ---
    n_out = sd["head"].shape[0]
    text_dim = (sd["text_head.1.weight"].shape[0]
                if "text_head.1.weight" in sd else 0)
    sk = "study_enc.seq_emb.weight"
    n_seq = sd[sk].shape[0] if sk in sd else 0
    pk = "series_enc.pos"
    n_pos = sd[pk].shape[1] if pk in sd else 0
    n_pos_types = sd[pk].shape[0] if pk in sd else 0
    senc = ("lstm" if "series_enc.rnn.weight_ih_l0" in sd
            else "tr" if any(k.startswith("series_enc.tr.") for k in sd)
            else "none")
    # NEW-style report aux (a5 v2 wave) carries aux_w/aux_b/aux_queries; the
    # OLD one (raux v1, which SHIPS) is a Sequential under report_aux.* and is
    # attached after construction - see attach_legacy_report_aux below.
    n_raux = sd["aux_w"].shape[0] if "aux_w" in sd else 0
    mkb = []
    while f"moaks_heads.{len(mkb)}.weight" in sd:
        mkb.append(sd[f"moaks_heads.{len(mkb)}.weight"].shape[0])

    # --- resolved from ck["args"], train.py default as fallback -------------
    r = {k: g(k, d) for k, d in TRAIN_DEFAULTS.items()}
    # ev2sfix --img-norm (scratch/ev2sfix_20261014/KERNEL_PORT.md): a NON-PERSISTENT input-normalisation buffer
    # with no weight signature. Resolved from args (absent/None -> imagenet); any value KneeNet does not implement
    # refuses here (refuse-not-skip) rather than silently building ImageNet stats.
    _imn = g("img_norm", "imagenet")
    if _imn is not None and str(_imn) not in ("imagenet", "half"):
        raise ValueError(f"ck['args']['img_norm']={_imn!r} not in ('imagenet', 'half'); refusing to build")
    # argparse writes None for some absent old keys; treat that as the default.
    for k, d in TRAIN_DEFAULTS.items():
        if r[k] is None:
            r[k] = d
    # band_pool and cell_attn also have an unambiguous weight signature; the
    # args value wins, the sniff covers checkpoints whose args predate the flag.
    band_pool = bool(r["band_pool"]) or ("band_proj.weight" in sd)
    cell_attn = int(r["cell_attn"] or 0) or (
        int(round(sd["study_enc.cell_pos"].shape[1] ** 0.5))
        if "study_enc.cell_pos" in sd else 0)
    flat_attn = bool(r["flat_attn"]) or ("study_enc.flat_gate" in sd)
    flat_queries = (sd["study_enc.flat_qmask"].nonzero().flatten().tolist()
                    if "study_enc.flat_qmask" in sd else None)

    # --bypass-study-transformer: args is the contract, the weights are the
    # cross-check. A study_enc-less model (evidence_mil) has neither.
    bypass_study_tr = bool(r["bypass_study_transformer"])
    has_study_enc = any(k.startswith("study_enc.") for k in sd)
    has_tr = any(k.startswith("study_enc.tr.") for k in sd)
    if has_study_enc and bypass_study_tr != (not has_tr):
        raise ValueError(
            "bypass_study_transformer mismatch: ck['args'] says "
            f"{bypass_study_tr} but the weights "
            f"{'carry' if has_tr else 'lack'} study_enc.tr.* keys. Refusing to "
            "rebuild a model the checkpoint does not describe.")

    # --- --pfrel: args is the contract; the weights are the cross-check -----
    pfrel = str(r["pfrel"] or "off")
    if pfrel not in ("off", "rel", "generic"):
        raise ValueError(f"ck['args']['pfrel']={pfrel!r}")
    has_pf = any(k.startswith("pfrel_net.") for k in sd)
    has_q = "pfrel_net.q" in sd
    has_g = "pfrel_net.gpool.v.weight" in sd
    if (pfrel != "off") != has_pf or (pfrel == "rel") != has_q \
            or (pfrel == "generic") != has_g:
        raise ValueError(
            f"pfrel mismatch: ck['args'] says {pfrel!r} but the weights "
            f"{'carry' if has_pf else 'lack'} pfrel_net.* "
            f"(queries={has_q}, generic pool={has_g}). Refusing to rebuild a "
            "model the checkpoint does not describe.")

    # --- --bonewsrpn: args is the contract; the weights are the cross-check -
    bone = str(r["bone"] or "off")
    if bone not in ("off", "roi", "dense"):
        raise ValueError(f"ck['args']['bone']={bone!r}")
    has_b = any(k.startswith("bone_net.") for k in sd)
    has_bq = "bone_net.q" in sd
    has_bd = "bone_net.dmlp.0.weight" in sd
    if (bone != "off") != has_b or (bone == "roi") != has_bq \
            or (bone == "dense") != has_bd:
        raise ValueError(
            f"bone mismatch: ck['args'] says {bone!r} but the weights "
            f"{'carry' if has_b else 'lack'} bone_net.* "
            f"(proposals={has_bq}, dense mlp={has_bd}). Refusing to rebuild a "
            "model the checkpoint does not describe.")

    _seq_adapt = int(r["seq_adapt"])
    _has_sa = "seq_adapt_mods.0.up" in sd
    if bool(_seq_adapt) != _has_sa or (_has_sa and sd["seq_adapt_mods.0.up"].shape[-1] != _seq_adapt):
        raise ValueError(f"seq_adapt mismatch: ck['args']['seq_adapt']={_seq_adapt} but the "
                         f"weights {'carry' if _has_sa else 'lack'} seq_adapt_mods.*. Refusing.")
    ps25 = bool(r["ps25"])
    if ps25 != ("ps25_head.2.weight" in sd):
        raise ValueError(f"ps25 mismatch: ck['args']['ps25']={ps25} but the weights "
                         f"{'carry' if 'ps25_head.2.weight' in sd else 'lack'} ps25_head.*. "
                         "Refusing to rebuild a model the checkpoint does not describe.")
    kw = dict(
        backbone=g("backbone"), pretrained=False, n_out=n_out,
        text_dim=text_dim, n_seq=n_seq, n_report_aux=n_raux,
        moaks_blocks=tuple(mkb),
        slice_enc=senc, n_slice_pos=n_pos, slice_pos_types=n_pos_types,
        slice_ctx=int(r["slice_ctx"]),
        img_norm=str(r["img_norm"]),
        stack_depth=bool(r["stack_depth"]),
        xfuse=kw_xfuse(r, sd),
        study_pool=str(r["study_pool"]),
        band_pool=band_pool,
        band_delta=float(r["band_delta"]),
        routing_prior=float(r["routing_prior"]),
        flat_attn=flat_attn, flat_queries=flat_queries,
        cell_attn=cell_attn,
        cell_stage=(-1 if r["cell_stage"] in (None, False)
                    else int(r["cell_stage"])),
        gate_init=float(r["gate_init"]),
        cell_slice_bands=int(r["cell_slice_bands"]),
        gem_p=float(r["gem_pool"]),
        osseous_factor=bool(r["osseous_factor"]),
        oa_gate=bool(r["oa_gate"]),
        query_family=str(r["query_family"] or "none"),
        # train.py: aux_seg_classes = 6 if --aux-seg-w > 0 else 0. The decoder
        # is training-only but its parameters must exist for strict=True
        # (unless a *_serve.pt with the decoder stripped is being read).
        aux_seg_classes=(6 if float(r["aux_seg_w"] or 0) > 0
                         and any(k.startswith("aux_seg_decoder.") for k in sd)
                         else 0),
        slice3d=str(r["slice3d"] or "none"),
        blurpool=int(r["blurpool"] or 0),
        qsa=str(r["qsa"] or "none"),
        qsa_gate_init=float(r["qsa_gate_init"]),
        drop_path=float(r["drop_path"] or 0.0),
        bypass_study_tr=bypass_study_tr,
        rubric_pool=kw_rubpool(r, sd),
        evidence_mil=bool(r["evidence_mil"]) or ("ev_R" in sd),
        evmil_prior=str(r["evmil_prior"] or "clinical"),
        tfcrop_types=kw_tfcrop(r, sd),
        evmil_tau_init=float(r["evmil_tau_init"] or 1.0),
        evmil_dif_init=str(r["evmil_dif_init"] or "mean"),
        evidence_mil_v2=kw_evmil_v2(r, sd),
        ev2_gamma_init=float(r["ev2_gamma_init"]),
        ev_adapter=kw_evadapter(r, sd),
        ev_adapter_k=int(r["ev_adapter_k"]),
        ev_adapter_width=int(r["ev_adapter_width"]),
        darc_local=bool(r["darc_local"]),
        syno_specialist=kw_synospec(r, sd),
        syno_gamma_init=float(r["syno_gamma_init"]),
        pfrel=pfrel,
        pfrel_lambda_pos=float(r["pfrel_lambda_pos"]),
        pfrel_beta=float(r["pfrel_beta"]),
        bone=bone,
        bone_k=int(r["bone_k"]),
        bone_tau=float(r["bone_tau"]),
        bone_shell=float(r["bone_shell"]),
        ps25=ps25,
        bn_valid_only=bool(r["bn_valid_only"]),
        seq_adapt=_seq_adapt,
    )
    nondefault = {k: r[k] for k, d in TRAIN_DEFAULTS.items() if r[k] != d}
    # Weight-sniffed overrides belong in the log line too.
    for k, v, d in (("bypass_study_transformer", bypass_study_tr, False),
                    ("rubric_pool", kw_rubpool(r, sd), False),
                    ("evidence_mil", kw_evmil(r, sd), False),
                    ("evidence_mil_v2", kw_evmil_v2(r, sd), "off"),
                    ("syno_specialist", kw_synospec(r, sd), False),
                    ("band_pool", band_pool, False),
                    ("cell_attn", cell_attn, 0),
                    ("flat_attn", flat_attn, False),
                    ("slice_enc", senc, "tr"),
                    ("n_out", n_out, 12),
                    ("n_report_aux", n_raux, 0),
                    ("moaks_blocks", tuple(mkb), ())):
        if v != d:
            nondefault[k] = v
    if "report_aux.0.weight" in sd:
        nondefault["report_aux"] = "v1_sequential"
    return kw, nondefault


def attach_legacy_report_aux(model, sd):
    """raux v1: a Sequential under report_aux.*, inert at inference.

    model.py builds the NEW aux head as aux_w/aux_b and keeps
    `self.report_aux = None` as a legacy attribute, so these five runs (raux,
    raux05, raux10, oaiaux, stage_twins/raux - one of them SHIPPING) cannot be
    strict-loaded without re-creating the module. test.csv has no Report
    column, so it is constructed purely to satisfy strict=True and never runs.
    """
    if "report_aux.0.weight" not in sd:
        return False
    dim = sd["report_aux.0.weight"].shape[0]
    model.report_aux = nn.Sequential(
        nn.LayerNorm(dim), nn.Dropout(0.1),
        nn.Linear(dim, sd["report_aux.2.weight"].shape[0]))
    return True


def build_from_checkpoint(path, device=None, ck=None, verbose=True,
                          syno_ablate=None):
    """path -> (model.eval() on device, ck["args"], meta).

    meta: {"ck", "sd", "flags" (non-default kwargs), "n_out", "legacy_raux"}.
    Raises on anything that would make the rebuilt model differ from the
    trained one. There is no skip path by design.
    """
    if ck is None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    # --srchead: args is the contract, srchead_lin.* the cross-check.
    _sh = _getter(a)("srchead", "") or ""
    _shk = sorted(k for k in sd if k.startswith("srchead_lin."))
    if bool(_sh) != bool(_shk) or (_sh and _shk != ["srchead_lin.bias", "srchead_lin.weight"]):
        raise ValueError(f"srchead mismatch: ck['args']['srchead']={_sh!r} but weights carry {_shk}. Refusing.")
    srchead_oai = None
    if _sh:
        sd = dict(sd)
        srchead_oai = {k: sd.pop(k) for k in _shk}
    kw, nondefault = resolve_flags(sd, a)
    model = KneeNet(**kw)
    # --trauma-head (scratch/traumahead): args is the contract, trauma_k_* weights the cross-check (refuse on disagreement).
    _th = bool(str(_getter(a)("trauma_head", False)) == "True" or _getter(a)("trauma_head", False) is True)
    if _th != ("trauma_k_lin.weight" in sd):
        raise ValueError(f"trauma_head mismatch: ck['args']['trauma_head']={_th} but the weights "
                         f"{'carry' if 'trauma_k_lin.weight' in sd else 'lack'} trauma_k_*. Refusing.")
    if _th:
        model._enable_trauma_head()
    # Evaluation-only decomposition switch for --syno-specialist. Refuse-not-
    # skip: asking to ablate a branch of a head this checkpoint does not have is
    # a silent no-op otherwise, which is how a decomposition table ends up
    # printing five identical rows.
    if syno_ablate is not None and str(syno_ablate) != "none":
        if not kw["syno_specialist"]:
            raise ValueError(
                f"--syno-ablate={syno_ablate} but {os.path.basename(path)} was "
                f"not trained with --syno-specialist (no syno_* weights)")
        model.syno_ablate = str(syno_ablate)
    legacy = attach_legacy_report_aux(model, sd)
    # strict=True: a missing or unexpected key means this is not the trained
    # model. It must abort the job; a silently short npz is how unionpack was
    # decided on 14 of its 18 members.
    model.load_state_dict(sd, strict=True)
    if device is not None:
        model = model.to(device)
    model.eval()
    if verbose:
        flags = " ".join(f"{k}={v}" for k, v in sorted(nondefault.items()))
        _g = _getter(a)
        _to = " ".join(f"{k}={_g(k, d)}" for k, d in sorted(
            TRAIN_ONLY_FLAGS.items()) if _g(k, d) not in (d, None))
        print(f"[build] {os.path.basename(path)} "
              f"backbone={kw['backbone']} "
              f"flags[{len(nondefault)}]: {flags or '(all default)'}"
              + (f" | train-only: {_to}" if _to else ""),
              flush=True)
    return model, a, {"ck": ck, "sd": sd, "flags": nondefault,
                      "n_out": kw["n_out"], "legacy_raux": legacy,
                      "srchead_oai": srchead_oai}
