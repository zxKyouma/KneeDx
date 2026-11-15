"""Hierarchical study-level model: slice -> series -> study -> 12 findings.

Three levels, because that is how the data is actually organised:

1. Slice encoder - a 2D ImageNet-pretrained backbone. With 4,407 studies and
   noisy labels, pretrained 2D features are worth far more than a 3D network
   trained from scratch on this much data. Adjacent slices are stacked into the
   three input channels so each forward pass still sees local through-plane
   context.
2. Series encoder - a small transformer over the slice sequence plus attention
   pooling. Findings are focal: a meniscal tear shows on a couple of slices, so
   mean pooling would dilute it while attention can concentrate on them.
3. Study encoder - a transformer over series embeddings, with a learned
   embedding for the series type (plane x fluid-sensitive) added in, so the
   model knows a coronal fat-sat series is a coronal fat-sat series rather than
   having to infer it from pixels.
"""

import math
import os
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# Finding families, indexed in the FINDINGS order:
#   ACL, MCL, MedMen, LatMen, MedOA, LatOA, PFOA, Effusion, Synovitis,
#   Baker's, Contusion, Fracture
# Clinical grouping: ligament / meniscus / osteoarthritis / inflammation /
# cystic / bone. Each group pairs a well-measured finding with a weak one,
# which is the whole hypothesis.
FAMILY_CLINICAL = [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5]

# The placebo. SAME number of families and the SAME family-size profile
# (2,2,3,2,1,2), so parameter count and group structure are identical, but no
# clinically related pair shares a family - verified in the unit check:
# (ACL,MCL), (MedMen,LatMen), the three OA compartments, (Effusion,Synovitis)
# and (Contusion,Fracture) are all split apart. Fixed and committed rather
# than seeded at runtime, so the control is reproducible.
FAMILY_RANDOM = [0, 1, 2, 3, 0, 2, 4, 1, 3, 5, 2, 5]

# --- MRI-CORE -------------------------------------------------------------
# ViT-B/16 pretrained with DINOv2-style self-distillation on MRI (Apache 2.0).
# It is not a timm model name, so it is built as a plain vit_base_patch16 and
# the published weights are remapped in. Three format details, all load-bearing:
#
#  * DINOv2 stores blocks CHUNKED as `blocks.{chunk}.{i}`. The inner index is
#    already the GLOBAL block index - chunk 0 holds 0-2, chunk 1 holds 3-5, and
#    so on - so the chunk level is pure padding and the remap drops it. Reading
#    the inner index as chunk-local would silently stack four copies of blocks
#    0-2 and discard the other nine.
#  * These weights carry NO LayerScale, so timm must be built with
#    init_values=None. With it set, ls1/ls2 appear in the model, never get
#    loaded, and stay at their init - a quiet 12-layer perturbation.
#  * `dynamic_img_size` interpolates the 224px position embeddings to whatever
#    input size arrives, so the arm is not locked to 224.
#
# Verified by execution at both 224 and 384: 150 tensors, zero missing, zero
# unexpected.
MRI_CORE_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".hf", "mri_core", "mri_core.pth")


def _adapt_in_chans(w, want):
    """timm's stem adaptation, done by hand because we load the weights
    ourselves rather than through create_model(pretrained=True). Sum the RGB
    stem down to one channel, tile it, and rescale so activation magnitude is
    preserved."""
    if w.shape[1] == want:
        return w
    return w.sum(1, keepdim=True).repeat(1, want, 1, 1) * (w.shape[1] / want)


def mri_core_state_dict(path, slice_ctx=3):
    """MRI-CORE checkpoint -> a timm vit_base_patch16 state dict."""
    raw = torch.load(path, map_location="cpu")
    sd = raw.get("teacher", raw) if isinstance(raw, dict) else raw
    out = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue                       # dino_head: the SSL projector
        k = k[len("backbone."):]
        if k == "mask_token":
            continue                       # DINOv2-only, no timm counterpart
        m = re.match(r"blocks\.\d+\.(\d+\..+)$", k)
        if m:
            k = "blocks." + m.group(1)
        out[k] = v
    out["patch_embed.proj.weight"] = _adapt_in_chans(
        out["patch_embed.proj.weight"], slice_ctx)
    return out


ORTHO_L_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "orthofoundation", "OrthoFoudation-L.pth")


def ortho_l_state_dict(path, slice_ctx=3):
    """OrthoFoundation-L checkpoint -> a timm vit_large_patch16_dinov3_qkvb
    state dict (scratch/orthofm/PREREG.md).

    The upstream file is Meta's reference-dinov3 naming, not timm's. Verified
    by execution on the 2026-09-14 download (sha256 385a7758...): 368 tensors,
    all float32, `backbone.` prefix, 24 blocks, embed dim 1024, patch 16,
    4 storage (register) tokens. The remap is mechanical and total:
      backbone.<k>            -> <k>
      ls1.gamma / ls2.gamma   -> gamma_1 / gamma_2
      storage_tokens          -> reg_token
      rope_embed.periods      -> rope.periods   (NON-persistent timm buffer, so it
                                is copied in by hand, not via load_state_dict;
                                the stored copy is a
                                bf16-rounded version of timm's own geometric
                                series - loaded anyway so the trunk is exactly
                                what was pretrained)
      attn.qkv.bias (3D)      -> attn.q_bias + attn.v_bias  (the k third is
                                EXACTLY 0.0 in every one of the 24 blocks,
                                which is what timm's non-persistent
                                attn.k_bias buffer already holds, so nothing
                                is lost)
      attn.qkv.bias_mask      -> dropped (the mask that zeroes the k third)
      mask_token              -> dropped (SSL-only, no timm counterpart)
    After the remap the load is strict-clean: zero missing, zero unexpected.
    """
    raw = torch.load(path, map_location="cpu")
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    out = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        if k == "mask_token" or k.endswith("attn.qkv.bias_mask"):
            continue
        if k == "storage_tokens":
            k = "reg_token"
        elif k == "rope_embed.periods":
            k = "rope.periods"
        elif k.endswith("ls1.gamma"):
            k = k[:-len("ls1.gamma")] + "gamma_1"
        elif k.endswith("ls2.gamma"):
            k = k[:-len("ls2.gamma")] + "gamma_2"
        elif k.endswith("attn.qkv.bias"):
            d = v.numel() // 3
            if float(v[d:2 * d].abs().max()) != 0.0:
                raise SystemExit("ortho_l: non-zero k bias in " + k +
                                 " - timm's k_bias buffer is a zero constant, "
                                 "so this checkpoint cannot be loaded losslessly")
            out[k[:-len("qkv.bias")] + "q_bias"] = v[:d]
            out[k[:-len("qkv.bias")] + "v_bias"] = v[2 * d:]
            continue
        out[k] = v
    out["patch_embed.proj.weight"] = _adapt_in_chans(
        out["patch_embed.proj.weight"], slice_ctx)
    return out


def build_backbone(name, pretrained, slice_ctx, drop_path=0.0):
    """timm.create_model, plus the local-weights backbones timm cannot name.

    drop_path (scratch/consttail/PREREG.md, C3): timm drop_path_rate for the
    timm backbones. 0.0 (default) adds NO kwarg at all, so every pre-existing
    arm is built by the exact same call as before; > 0 is training-only
    (identity at eval, no state_dict change).
    """
    if name == "r3d_18":
        # scratch/basefam/PREREG.md arm V: native 3D CNN, torchvision r3d_18,
        # Kinetics-400 init. NOT a timm model, so it is built and returned
        # before the timm branch below runs (existing arms are untouched -
        # this is a new, additive name, never reached by any pre-existing
        # --backbone value). slice_ctx is unused here by design: the wrapper
        # takes whatever (N, Z, H, W) --stack-depth's reshape produces and
        # maps Z straight onto r3d_18's depth axis; KneeNet.__init__ enforces
        # that this backbone is only ever built with stack_depth=True.
        if drop_path > 0:
            raise ValueError("--drop-path is only wired for timm backbones")
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        net = r3d_18(weights=weights)
        net.fc = nn.Identity()
        return R3D18Backbone(net)
    if name == "qwen35v_2b":
        # scratch/qwenvit/PREREG.md: Qwen3.5-2B vision tower as the slice
        # encoder (src/qwenvit.py). NEW, additive name - no pre-existing
        # --backbone value reaches this branch. pretrained=False (inference /
        # ckpt_build) builds from the vendored config with no weight file.
        if slice_ctx != 3 or drop_path > 0:
            raise ValueError("qwen35v_2b needs slice_ctx == 3 and no --drop-path")
        from qwenvit import QwenVisionSlice
        return QwenVisionSlice(pretrained=pretrained)
    import timm
    _dp = {"drop_path_rate": float(drop_path)} if drop_path > 0 else {}
    if name == "dinov3_vitl_ortho":
        # scratch/orthofm/PREREG.md: OrthoFoundation-L, a DINOv3 ViT-L/16 trunk
        # continued-pretrained on 894k knee MRI slices + 358k knee radiographs
        # (https://github.com/ytrsk/OrthoFoundation). NEW, additive name - no
        # pre-existing --backbone value reaches this branch, so every existing
        # arm is byte-identical to before. Same shape as the mri_core branch
        # below: build the timm architecture with pretrained=False (so the
        # inference path never needs the upstream file) and load our local
        # weights over it during training only.
        m = timm.create_model("vit_large_patch16_dinov3_qkvb", pretrained=False,
                              num_classes=0, in_chans=slice_ctx,
                              dynamic_img_size=True, **_dp)
        if not pretrained:
            return m
        path = os.environ.get("ORTHO_L_CKPT", ORTHO_L_DEFAULT)
        if not os.path.exists(path):
            raise SystemExit(f"dinov3_vitl_ortho: no weights at {path} "
                             f"(set ORTHO_L_CKPT)")
        sd = ortho_l_state_dict(path, slice_ctx)
        # timm registers rope.periods as a NON-persistent buffer, so it is not
        # in state_dict() and strict accounting would report it as unexpected.
        # Copy it in by hand instead of dropping it, so the trunk runs on the
        # exact periods it was pretrained with (the stored copy is a
        # bf16-rounded version of timm's own fp32 geometric series).
        _per = sd.pop("rope.periods", None)
        if _per is not None and hasattr(m, "rope"):
            with torch.no_grad():
                m.rope.periods.copy_(_per.to(m.rope.periods.dtype))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"dinov3_vitl_ortho: key mismatch after remap - "
                             f"missing={missing[:8]} unexpected={unexpected[:8]}")
        print(f"[ortho_l] loaded {len(sd)} pretrained tensors from {path}",
              flush=True)
        return m
    if name == "mri_core" and drop_path > 0:
        raise ValueError("--drop-path is only wired for timm backbones")
    if name != "mri_core":
        # ViT-family backbones bake the input resolution into their position
        # embeddings - DINOv2 ViT-S/14 defaults to 518px, so creating it and
        # feeding 224px silently mismatches or raises. dynamic_img_size makes
        # timm interpolate the position embedding at forward time from the
        # actual input, which is the same mechanism the mri_core path below
        # already relies on.
        #
        # Tried-then-fallback rather than a name test, because the set of
        # backbones accepting this kwarg is a timm implementation detail and a
        # name pattern would silently stop matching on a timm upgrade. ConvNeXt
        # and friends raise TypeError on the unknown kwarg and take the second
        # branch, so every existing arm is bit-identical to before this change.
        try:
            return timm.create_model(name, pretrained=pretrained,
                                     num_classes=0, in_chans=slice_ctx,
                                     dynamic_img_size=True, **_dp)
        except TypeError:
            pass
        return timm.create_model(name, pretrained=pretrained,
                                 num_classes=0, in_chans=slice_ctx, **_dp)
    m = timm.create_model("vit_base_patch16_224", pretrained=False,
                          num_classes=0, in_chans=slice_ctx,
                          init_values=None, dynamic_img_size=True)
    # pretrained=False is the INFERENCE path: the notebook builds the
    # architecture and then loads our own trained checkpoint over it, so the
    # MRI-CORE file must not be required at scoring time.
    if not pretrained:
        return m
    path = os.environ.get("MRI_CORE_CKPT", MRI_CORE_DEFAULT)
    if not os.path.exists(path):
        raise SystemExit(f"mri_core backbone: no weights at {path} "
                         f"(set MRI_CORE_CKPT)")
    sd = mri_core_state_dict(path, slice_ctx)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # Loud, because a silently-random ViT still trains to a plausible-looking
    # curve and would be read as an architecture verdict rather than a bug.
    if missing or unexpected:
        raise SystemExit(f"mri_core: key mismatch after remap - "
                         f"missing={missing[:8]} unexpected={unexpected[:8]}")
    print(f"[mri_core] loaded {len(sd)} pretrained tensors from {path}",
          flush=True)
    return m


class R3D18Backbone(nn.Module):
    """Wraps torchvision's r3d_18 (native 3D CNN, Kinetics-400 init) so it
    slots into KneeNet.backbone via the existing --stack-depth path
    (scratch/basefam/PREREG.md arm V). --stack-depth already reshapes the
    depth axis into the backbone call's input as (B*S, Z, H, W) with
    Z == slice_ctx, and (with band_pool/cell_attn/aux_seg all off, the
    default gem_p==1.0 branch) calls `self.backbone(x)` directly - so this
    wrapper's forward signature is exactly that: (N, Z, H, W) in, (N, 512)
    out. It is not built to answer forward_features/forward_head (the
    cell-attn/band-pool/aux-seg branches); arm V does not use those flags.

    r3d_18 is a Conv3d stack expecting (N, C, D, H, W) with C=3 (Kinetics
    RGB stem). Z (24 adjacent through-plane slices, --slice-ctx 24
    --n-slices 24) maps straight onto D with no reshuffling - a channel axis
    is inserted at dim 1 and simply repeated 3x so the pretrained 3-channel
    stem sees three identical copies (no learned stem surgery, same
    normalisation KneeNet already applies to any slice_ctx != 3 backbone:
    per-channel mean/std 0.449/0.226, i.e. this wrapper receives already-
    normalised input).
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.num_features = 512          # r3d_18's fc input width

    def forward(self, x):
        # x: (N, Z, H, W) -> (N, 1, Z, H, W) -> (N, 3, Z, H, W)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        return self.net(x)


class Slice3DConvDW(nn.Module):
    """--slice3d (scratch/slice3d/PREREG.md): through-plane feature mixing
    INSIDE the ConvNeXt encoder. Wraps one timm ConvNeXtBlock.conv_dw (the
    7x7 depthwise conv at the head of every block). The backbone receives
    slices flattened into the batch dim, (B*S*Z, k, H, W) with Z adjacent
    in the LAST factor (KneeNet._stack_ctx reshapes (B*S, Z, ...) ->
    (B*S*Z, ...), and src/data.py samples idx monotone in z, so consecutive
    rows are consecutive slices). KneeNet.forward stamps the per-forward
    slice count on `self.S` before the backbone call; the wrapper then
    recovers (N/S, S, C, H, W), mixes along S with ZERO padding at both
    ends, and returns (N, C, H, W) so the rest of the block is untouched.

    mode="shift": zero-parameter temporal shift (TSM). Channels [0, C/8)
      of slice s are replaced by those of slice s-1, channels [C/8, C/4)
      by those of slice s+1 (zeros at the ends); the rest pass through.
      Then the ORIGINAL conv_dw runs. Same param count, same FLOPs.
    mode="inflate": I3D-style centre-initialised inflation of the 7x7
      depthwise conv to 3x7x7, computed as the ORIGINAL 2D conv (pretrained
      weight + bias, unchanged) PLUS two NEW zero-initialised depthwise 7x7
      convs (no bias) on the s-1 and s+1 slices, summed. Mathematically the
      inflated Conv3d, on the fast cuDNN 2D path. At init the two taps
      contribute exact zeros, so the output equals the 2D model bit-for-bit;
      they receive nonzero input so zero-init is trainable (not a gate).

    Contract: this wrapper is installed ONLY when slice3d != "none", so the
    default model is the pre-existing module graph by construction (no
    wrapper, no extra op, identical state_dict keys). With the flag on, the
    original conv lives at <block>.conv_dw.conv.* and the inflate taps at
    <block>.conv_dw.prev.weight / .next.weight - all under "backbone.", so
    they fall into the backbone LR-mult / weight-decay group and the EMA
    shadow (state_dict-based) automatically. Consumers rebuilding a model
    from a checkpoint MUST pass slice3d=ck["args"].get("slice3d", "none").
    """

    def __init__(self, conv, mode):
        super().__init__()
        if mode not in ("shift", "inflate"):
            raise ValueError(f"slice3d mode {mode!r}")
        if not isinstance(conv, nn.Conv2d) or conv.stride != (1, 1):
            raise ValueError(f"--slice3d expects a stride-1 nn.Conv2d "
                             f"depthwise conv, got {conv}")
        self.mode = mode
        self.conv = conv
        self.S = 0                      # stamped per forward by KneeNet
        if mode == "inflate":
            C = conv.in_channels
            mk = lambda: nn.Conv2d(C, conv.out_channels, conv.kernel_size,
                                   stride=1, padding=conv.padding,
                                   dilation=conv.dilation, groups=conv.groups,
                                   bias=False)
            # fork_rng: nn.Conv2d's default init draws from the global RNG,
            # which would shift the random init of every module built AFTER
            # the backbone (series/study encoders, heads) relative to the
            # flag-off model under the same seed. The taps are zeroed anyway,
            # so build them RNG-neutrally; check_identity.py relies on this.
            with torch.random.fork_rng(devices=[]):
                self.prev = mk()
                self.next = mk()
            nn.init.zeros_(self.prev.weight)
            nn.init.zeros_(self.next.weight)

    def _split(self, x):
        N, C, H, W = x.shape
        S = int(self.S)
        if S <= 0 or N % S:
            raise RuntimeError(f"--slice3d: slice count {S} not stamped or "
                               f"does not divide batch {N}")
        return x.reshape(N // S, S, C, H, W)

    def forward(self, x):
        N, C, H, W = x.shape
        x5 = self._split(x)
        z = x5[:, :1]
        if self.mode == "shift":
            f = C // 8
            a = x5[:, :, :f]              # take from s-1
            b = x5[:, :, f:2 * f]         # take from s+1
            a = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)
            b = torch.cat([b[:, 1:], torch.zeros_like(b[:, :1])], dim=1)
            y = torch.cat([a, b, x5[:, :, 2 * f:]], dim=2).reshape(N, C, H, W)
            return self.conv(y)
        # inflate: centre tap = original conv on x (bit-identical), plus
        # zero-init taps on the s-1 / s+1 slices (zero-padded at the ends).
        xm = torch.cat([torch.zeros_like(z), x5[:, :-1]], dim=1).reshape(N, C, H, W)
        xp = torch.cat([x5[:, 1:], torch.zeros_like(z)], dim=1).reshape(N, C, H, W)
        return self.conv(x) + self.prev(xm) + self.next(xp)


def install_slice3d(backbone, mode):
    """Replace every timm ConvNeXtBlock.conv_dw in `backbone` with a
    Slice3DConvDW(mode). Returns the list of wrappers (KneeNet stamps S on
    them each forward). Raises if the backbone has no ConvNeXt blocks, so a
    mis-typed --backbone fails at build time rather than silently running
    the 2D model under a 3D tag."""
    from timm.models.convnext import ConvNeXtBlock
    mods = []
    for blk in backbone.modules():
        if isinstance(blk, ConvNeXtBlock):
            blk.conv_dw = Slice3DConvDW(blk.conv_dw, mode)
            mods.append(blk.conv_dw)
    if not mods:
        raise ValueError(f"--slice3d {mode}: no ConvNeXtBlock in "
                         f"{type(backbone).__name__}")
    return mods


class BlurConv2d(nn.Conv2d):
    """--blurpool (scratch/regbundle/PREREG.md): anti-aliased downsampling
    (Zhang 2019 BlurPool, ELNet's MRI use). A FIXED, parameter-free binomial
    [1,2,1]x[1,2,1]/16 depthwise blur (stride 1, reflect pad, same size) is
    applied to the INPUT of a strided conv, then the ORIGINAL conv runs
    unchanged. Installed by __class__ swap on the existing nn.Conv2d
    (install_blurpool), so the parameter tensors, their state_dict keys and
    the pretrained weights are untouched; the filter is a NON-persistent
    buffer, so the state_dict is byte-identical to the unblurred model and
    a consumer cannot sniff the flag from the weights - it must read
    ck["args"]["blurpool"] (osseous_factor / slice3d precedent). Zero new
    parameters; the blur is the only extra op and runs at train AND serve.
    """

    def forward(self, x):
        f = self.blur_filt
        if f.device != x.device or f.dtype != x.dtype:
            f = f.to(device=x.device, dtype=x.dtype)
        C = x.shape[1]
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        x = F.conv2d(x, f.expand(C, 1, 3, 3), stride=1, groups=C)
        return super().forward(x)


def install_blurpool(backbone):
    """Swap every stride>1 downsampling conv of a timm ConvNeXt - the 4x4/4
    patch stem conv and the 2x2/2 conv inside each non-Identity
    stage.downsample - to BlurConv2d IN PLACE. Returns the list of swapped
    convs (convnext_tiny: 4 = stem + stages 1..3). Raises on a non-ConvNeXt
    backbone or an unexpected conv type, so a mis-typed --backbone fails at
    build time instead of silently running unblurred under the tag."""
    if not (hasattr(backbone, "stem") and hasattr(backbone, "stages")):
        raise ValueError(f"--blurpool: {type(backbone).__name__} has no "
                         f"ConvNeXt stem/stages")
    convs = [backbone.stem[0]]
    for st in backbone.stages:
        ds = getattr(st, "downsample", None)
        if isinstance(ds, nn.Sequential):
            convs.append(ds[-1])
    filt = torch.tensor([1.0, 2.0, 1.0])
    filt = (filt[:, None] * filt[None, :]) / 16.0
    for c in convs:
        if type(c) is not nn.Conv2d or c.stride[0] < 2:
            raise ValueError(f"--blurpool: expected a strided nn.Conv2d, got "
                             f"{type(c).__name__} stride={getattr(c, 'stride', None)}")
        c.__class__ = BlurConv2d
        c.register_buffer("blur_filt", filt.clone().view(1, 1, 3, 3),
                          persistent=False)
    return convs


class AuxSegHead(nn.Module):
    """scratch/auxdice/PREREG.md: light decoder off the encoder's last
    (N,C,Hf,Wf) feature map, predicting the 6-class anatomical pseudo-mask
    (background/femur_bone/femoral_cartilage/tibia_bone/medial_tibial_
    cartilage/lateral_tibial_cartilage, scratch/structruth/unet2d.py
    convention). 1x1 conv to cut channels, two nearest-upsample+conv3x3
    stages (Hf,Wf -> 4*Hf,4*Wf; 12x12 -> 48x48 for convnext_tiny/384),
    1x1 conv to class logits. Built ONLY when --aux-seg-w > 0 (aux_seg_classes
    > 0); discarded at serve time (never referenced by inference)."""

    def __init__(self, in_dim, n_classes, hidden=128):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, hidden, 1)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(hidden, n_classes, 1)

    def forward(self, fm):
        x = self.reduce(fm)
        x = self.up1(x)
        x = self.up2(x)
        return self.out(x)


class AttnPool(nn.Module):
    """Gated attention pooling (Ilse et al.), masked."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        # x: (B, N, D)
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))  # (B,N,1)
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        a = torch.softmax(a, dim=1)
        return (a * x).sum(1), a.squeeze(-1)


class SeriesEncoder(nn.Module):
    """Transformer over the slices of one series, then attention-pool.

    Without `n_pos` this is EXACTLY permutation-invariant over slice tokens -
    measured at 1.2e-07, i.e. float precision. A transformer with no positional
    encoding is permutation-equivariant and the attention pool that follows is
    permutation-invariant, so nothing downstream can know where in the stack a
    slice came from. Combined with the backbone's global spatial pool, which
    averages H and W away, the model sees WHAT textures appear somewhere in a
    series and nothing about WHERE.

    That matters because slice index is anatomy here. `preprocess.py` sorts by
    ImagePositionPatient projected on the slice normal and the dataset reverses
    sagittal stacks for right knees, so after canonicalisation the index means
    medial->lateral on sagittal, anterior->posterior on coronal, and
    superior->inferior on axial - consistently across studies. The model's only
    access to any of that today is the +/-1 neighbourhood smuggled in by the
    3-slice channel stack.

    `n_pos_types` gives each series type its own table, because position means
    a different axis per plane: a single shared ramp would have to average a
    medial-lateral gradient against an anterior-posterior one.

    Scale is set deliberately, not defaulted. Measured on a real study, slice
    tokens have std 1.09 and per-token norm 29.6, so nn.Embedding's default
    N(0,1) would arrive at 0.94x the token's own norm and drown the content -
    the mirror image of the seq_emb bug, where std=0.02 was 50x too SMALL and
    the feature was inert through a full 5.5-hour run. std=0.2 puts it at about
    a fifth of the token norm: enough to be used, not enough to swamp.
    """

    POS_STD = 0.2

    def __init__(self, dim, n_layers=2, n_heads=8, drop=0.1, n_pos=0,
                 n_pos_types=0, mode="tr"):
        super().__init__()
        # mode selects the slice-sequence mixer. "tr" is the shipped
        # transformer; "lstm" is the bi-LSTM the RSNA-2024 winners' ablation
        # measured ABOVE self-attention for slice sequences; "none" drops the
        # mixer entirely (field-style 2.5D: per-slice features + pooling).
        # Distinct member families, selected by --slice-enc and sniffed off
        # the state dict by every loader.
        self.mode = mode
        if mode == "tr":
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
                dropout=drop, batch_first=True, norm_first=True, activation="gelu")
            self.tr = nn.TransformerEncoder(layer, n_layers)
        elif mode == "lstm":
            self.rnn = nn.LSTM(dim, dim // 2, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=drop)
        elif mode != "none":
            raise ValueError(f"unknown slice-enc mode {mode!r}")
        self.pool = AttnPool(dim)
        self.n_pos_types = n_pos_types
        self.pos = (nn.Parameter(torch.randn(max(n_pos_types, 1), n_pos, dim)
                                 * self.POS_STD) if n_pos else None)

    def forward(self, x, stype=None, return_tokens=False):  # (B*S, Z, D), (B*S,)
        if self.pos is not None:
            Z = x.shape[1]
            if Z != self.pos.shape[1]:
                raise ValueError(
                    f"slice-position table has {self.pos.shape[1]} entries but "
                    f"got {Z} slices. The table is tied to --n-slices, so a "
                    f"checkpoint cannot be run at a different slice count.")
            if self.n_pos_types and stype is not None:
                x = x + self.pos[stype.clamp(0, self.n_pos_types - 1)]
            else:
                x = x + self.pos[0]
        if self.mode == "tr":
            x = self.tr(x)
        elif self.mode == "lstm":
            x, _ = self.rnn(x)
        pooled, _attn = self.pool(x)
        # Read-only instrumentation for scratch/critbottle. The per-slice
        # attention is already computed and discarded here; stashing it changes
        # no weight and no return value, so predictions stay bit-identical.
        if getattr(self, "want_attn", False):
            self._attn = _attn.detach()
        if getattr(self, "want_tokens", False):
            # per-slice tokens (B*S, Z, D) - the input the criterion bottleneck
            # needs. Already computed; stashing changes no return value.
            self._tokens = x.detach()
        return (pooled, x) if return_tokens else pooled


class StudyEncoder(nn.Module):
    """Per-finding routing over series.

    A single pooled study vector forces all twelve findings to share one view of
    the study, but they do not live in the same series: a meniscal tear reads on
    sagittal PD, MCL on coronal, effusion and synovitis on fluid-sensitive.
    Instead of one attention pool, each finding gets a learned query that
    cross-attends over the series embeddings, so it can weight the series it
    actually needs. Returns (B, n_out, D) - one vector per finding.
    """

    # Fixed anatomical routing prior, (finding, series_type) with types ordered
    # plane*2+fluid: Sag Sag-FS Cor Cor-FS Ax Ax-FS. 1 marks the series types a
    # radiologist would read that finding on. This exists for the findings whose
    # labels are too noisy to teach routing (Synovitis KL 0.121, the flattest of
    # all twelve): a prior injects what the loss cannot supply. Applied as an
    # additive attention-logit tilt scaled by routing_prior_w - a buffer-like
    # CONSTANT, deliberately not a Parameter and not in state_dict, so existing
    # checkpoints load strict and w=0 is bit-identical to no prior at all.
    ROUTE_PRIOR = [
        # Sag  SagFS  Cor  CorFS  Ax  AxFS
        [1.0,  1.0,   0.0, 0.0,   0.0, 0.0],   # ACL          - sagittal
        [0.0,  0.0,   1.0, 1.0,   0.0, 0.0],   # MCL          - coronal
        [1.0,  1.0,   0.0, 0.0,   0.0, 0.0],   # Med Meniscus - sagittal
        [1.0,  1.0,   0.0, 0.0,   0.0, 0.0],   # Lat Meniscus - sagittal
        [0.0,  0.0,   1.0, 1.0,   0.0, 0.0],   # Medial OA    - coronal
        [0.0,  0.0,   1.0, 1.0,   0.0, 0.0],   # Lateral OA   - coronal
        [0.0,  0.0,   0.0, 0.0,   1.0, 1.0],   # PF OA        - axial
        [0.0,  0.0,   0.0, 0.0,   0.0, 1.0],   # Effusion     - axial FS
        [0.0,  0.0,   0.0, 0.0,   0.0, 1.0],   # Synovitis    - axial FS
        [0.0,  0.0,   0.0, 0.0,   0.0, 1.0],   # Baker's      - axial FS
        [0.0,  1.0,   0.0, 1.0,   0.0, 1.0],   # Contusion    - any fat-sat
        [0.0,  1.0,   0.0, 1.0,   0.0, 1.0],   # Fracture     - any fat-sat
    ]

    def __init__(self, dim, n_types=6, n_out=12, n_layers=2, n_heads=8, drop=0.1,
                 n_seq=0, routing_prior=0.0, flat_attn=False, cell_attn=0,
                 gate_init=0.1, cell_slice_bands=3, n_aux=0,
                 query_family="none", qsa="none", qsa_gate_init=0.1,
                 flat_queries=None, study_pool="route",
                 bypass_study_tr=False):
        super().__init__()
        self.study_pool = study_pool
        # --bypass-study-transformer (scratch/evmil_bypass/PREREG.md): the
        # attribution control for EvidenceMIL. EVERYTHING else is unchanged --
        # slice attention pool, series/type/sequence embeddings, the 12
        # cross-attention queries, the heads and the train-only aux head --
        # and ONLY the inter-series self-attention/FFN stack is removed, so
        # series tokens reach the queries untransformed. self.tr is set to
        # None rather than run as identity: the key set then differs loudly
        # (no study_enc.tr.*), so a checkpoint built with the wrong value of
        # the flag cannot strict-load, and ckpt_build refuses either way.
        self.bypass_study_tr = bool(bypass_study_tr)
        if self.bypass_study_tr and study_pool == "mean":
            raise ValueError("--bypass-study-transformer and --study-pool mean "
                             "both remove the study transformer; refusing an "
                             "ambiguous configuration.")
        if study_pool == "mean" and (routing_prior or n_aux or flat_attn
                                     or cell_attn or query_family not in ("none", "")
                                     or qsa not in ("none", "")):
            raise ValueError(
                "--study-pool mean bypasses the study transformer and the "
                "per-finding queries, so routing-prior / report-aux / "
                "flat-attn / cell-attn / query-family / qsa have nothing to "
                "act on; refusing to run a configuration where those flags "
                "would be silently ignored.")
        self.routing_prior_w = float(routing_prior)
        self.n_heads = n_heads
        if self.routing_prior_w and (n_out != 12 or n_types != 6):
            raise ValueError("routing prior table is defined for 12 findings x "
                             f"6 series types, got {n_out} x {n_types}")
        if self.routing_prior_w and n_aux:
            # The tilt is (n_out, n_types); with extra queries it would silently
            # broadcast against the wrong axis rather than fail.
            raise ValueError("--routing-prior and report-aux queries are not "
                             "compatible: the prior table has no rows for the "
                             "auxiliary queries.")
        self._route_prior = torch.tensor(self.ROUTE_PRIOR)
        self.type_emb = nn.Embedding(n_types, dim)
        # Pulse-sequence weighting, as a SECOND embedding rather than a wider
        # composite id. (plane, fluid) x weighting would be 30 sparse tokens
        # sharing no statistics; summing two embeddings lets "coronal" and "T1"
        # each be learned from every series that carries them. n_seq=0 leaves
        # the module absent entirely, so checkpoints trained without it load
        # with strict=True and inference can tell the two apart by key.
        self.seq_emb = nn.Embedding(n_seq, dim) if n_seq else None
        if self.seq_emb is not None:
            # Match type_emb's scale. nn.Embedding defaults to N(0,1), giving
            # row norms ~27 at dim=768; an earlier version used std=0.02 here,
            # which is ~50x smaller, and since both are summed into the SAME
            # residual stream the sequence signal was drowned by the series-type
            # one. Measured on a trained checkpoint: swapping every series to
            # "unknown" moved predictions by a mean of 0.0013, against a 0.03
            # noise band - the feature was inert, and the arm could not have
            # demonstrated its value either way.
            nn.init.normal_(self.seq_emb.weight, std=1.0)
            with torch.no_grad():
                # Index 0 is "unknown" (19.3% of series). Start it at zero so an
                # unreadable description contributes nothing at step 0; it is
                # free to learn away from zero afterwards, which it does.
                self.seq_emb.weight[0].zero_()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=drop, batch_first=True, norm_first=True, activation="gelu")
        self.tr = (None if self.bypass_study_tr
                   else nn.TransformerEncoder(layer, n_layers))
        self.queries = nn.Parameter(torch.randn(n_out, dim) * 0.02)
        # Auxiliary queries, one per unscored report target, kept as a SEPARATE
        # parameter rather than widening `queries`. Two reasons, both practical:
        # the scored path's shapes stay byte-compatible so every existing
        # checkpoint still loads, and report_aux is training-only (test.csv has
        # no Report column), so at inference n_aux=0 and this key is simply
        # absent.
        #
        # WHY IT EXISTS. The old head read `report_aux(study_emb.mean(1))` - the
        # MEAN over the twelve finding queries. Its gradient w.r.t. query i is
        # then (1/12)*W^T*g, IDENTICAL for every i, so eight auxiliary targets
        # could not teach any specific finding query anything. That is analytic,
        # not measured, and it is why the raux sweep came out null. Three of the
        # eight targets are compartment-localised cartilage grades that each
        # predict their own compartment best (Medial 0.910 / Lateral 0.809 /
        # PF 0.854), i.e. exactly the supervision Lateral OA and PF OA need -
        # and it has never reached them.
        self.aux_queries = (nn.Parameter(torch.randn(n_aux, dim) * 0.02)
                            if n_aux else None)
        # ---- finding-family embedding ------------------------------------
        # The twelve finding queries are independent vectors that share nothing
        # but the trunk. Findings are not independent: Contusion and Fracture
        # are the same tissue signal split by one feature (a cortical line),
        # Effusion and Synovitis are the same joint-fluid neighbourhood, and
        # the three OA compartments are one disease read in three places. A
        # learned per-family vector added to each query lets the well-measured
        # member of a family carry routing information to the weak one -
        # Fracture 0.932 -> Contusion 0.908, Effusion 0.968 -> Synovitis 0.872,
        # Medial OA 0.968 -> Lateral OA 0.908.
        #
        # `random` is the PLACEBO, and it is the point of the design: it adds
        # identical parameters and identical capacity with the clinical
        # structure destroyed. If clinical beats random the grouping carries
        # information; if both move together it was capacity, which is the
        # mechanism story this project has been fooled by before.
        self.query_family = query_family
        self.fam_emb = None
        if query_family and query_family != "none":
            groups = (FAMILY_CLINICAL if query_family == "clinical"
                      else FAMILY_RANDOM)
            if n_out != len(groups):
                raise ValueError(
                    f"--query-family needs {len(groups)} scored queries, "
                    f"got n_out={n_out}")
            self.register_buffer("_fam_idx", torch.tensor(groups),
                                 persistent=True)
            self.fam_emb = nn.Embedding(int(max(groups)) + 1, dim)
            nn.init.normal_(self.fam_emb.weight, std=0.02)
        self.route = nn.MultiheadAttention(dim, n_heads, dropout=drop,
                                           batch_first=True)
        self.norm = nn.LayerNorm(dim)
        # ---- inter-finding query self-attention (scratch/qsa/PREREG.md) ----
        # The scored queries cross-attend to series/slice tokens but never to
        # each other, so Synovitis cannot condition on what the Effusion query
        # found in THIS study; the only inter-finding link is the trunk (and
        # the static query_family bias, whose placebo won). ONE pre-norm MHA
        # block over the n_out scored queries, residual through a learned
        # scalar gate initialised SMALL BUT NON-ZERO (zero-init gates collapse
        # in this codebase: CLAUDE.md tripwire).
        #   post = applied to the ROUTED outputs, after every cross-attention
        #          path and before the heads: a study-dependent exchange (the
        #          Med-DGTN / label-correlation form). This is the mechanism
        #          arm.
        #   pre  = applied to the query PARAMETERS before cross-attention.
        #          q is the same constant for every study, so this is a
        #          static re-parametrisation of `queries` (no new function
        #          class, optimisation only) - kept as the equal-parameter
        #          capacity placebo for post, not as a mechanism arm.
        #   none = no module, no state_dict key, forward untouched
        #          (bit-identical to every pre-existing arm).
        # Aux report / MOAKS queries are EXCLUDED (scored rows only), for the
        # same reason fam_emb excludes them: a training-only object must not
        # change the scored path when report_aux is on. Not in state_dict
        # unless != none; consumers must rebuild from ck["args"]["qsa"] and
        # ck["args"]["qsa_gate_init"] (slice3d / blurpool precedent).
        self.qsa = str(qsa or "none")
        if self.qsa not in ("none", "pre", "post"):
            raise ValueError(f"qsa={self.qsa!r}")
        self.qsa_attn = self.qsa_norm = self.qsa_gate = None
        if self.qsa != "none":
            if float(qsa_gate_init) == 0.0:
                raise ValueError("--qsa-gate-init 0 is a zero-init gate; "
                                 "those collapse here (CLAUDE.md tripwire)")
            self.qsa_attn = nn.MultiheadAttention(dim, n_heads, dropout=drop,
                                                  batch_first=True)
            self.qsa_norm = nn.LayerNorm(dim)
            self.qsa_gate = nn.Parameter(
                torch.full((1,), float(qsa_gate_init)))
        # ---- flat cross-series slice routing -----------------------------
        # SeriesEncoder attention-pools Z away BEFORE the finding queries see
        # anything, so a query can only weight a SERIES - never a slice. That
        # is a real bottleneck for compartment-localised findings: the medial
        # meniscus reads on the medial-most few sagittal slices, and after the
        # pool those are averaged against the other twenty. It also explains
        # the routing anomaly this project measured and could not act on -
        # Medial Meniscus puts 0.007 of its mass on sagittal while scoring
        # 0.95, because picking the coronal SERIES is the only way it can
        # reach a vector that happens to carry the signal.
        #
        # This adds a second cross-attention whose keys are all S*Z slice
        # tokens, gated by a ZERO-init scalar and added as a residual to the
        # series-routed output. At step 0 the contribution is exactly 0, so
        # the model is bit-identical to the control and the flat path can only
        # be used if it earns its way off zero - the same zero-init residual
        # discipline band_proj uses.
        #
        # No slice-position table here on purpose: --slice-pos lost 9/9 epochs
        # as its own arm, and content-based selection is the half of that
        # hypothesis worth testing. Keeping position off makes this a single
        # variable.
        self.flat_route = (nn.MultiheadAttention(dim, n_heads, dropout=drop,
                                                 batch_first=True)
                           if flat_attn else None)
        self.flat_norm = nn.LayerNorm(dim) if flat_attn else None
        # gate_init is NOT 0. Measured on flatattn (job 59043054): a zero-init
        # gate gave flat-gate 0.0002 at ep0 and 0.0000 at ep1 - it COLLAPSED.
        # The mechanism: at gate exactly 0 the attention projection receives no
        # gradient at all (d out/d proj is proportional to the gate), so it stays
        # at its random init, and the gate is left deciding whether a RANDOM
        # projection of the tokens helps. It does not, so the gate shrinks to
        # zero and the branch never gets the chance to learn a useful one.
        # That makes a zero-init arm unable to distinguish "mechanism useless"
        # from "optimiser could not reach it" - so it cannot answer its own
        # question, which is the seq_emb failure in a new costume.
        # A small nonzero gate gives the projection real gradient from step 0.
        # Parity with the control becomes approximate rather than exact; that is
        # the price of an interpretable experiment.
        self.flat_gate = (nn.Parameter(torch.full((1,), float(gate_init)))
                          if flat_attn else None)
        # flat_queries (scratch/flatattn2/PREREG.md Amendment 2): restrict the
        # flat residual to a SUBSET of the scored queries (e.g. Contusion only)
        # so a single-finding branch can be tested without touching the other
        # eleven. Stored as a buffer so the state dict reveals it. None/all-ones
        # = bit-identical to the unrestricted flat_attn model.
        if flat_attn:
            qm = torch.ones(n_out + n_aux)
            if flat_queries is not None:
                qm.zero_(); qm[list(flat_queries)] = 1.0
            self.register_buffer("flat_qmask", qm)
        # ---- cell-level (spatial) routing ---------------------------------
        # flat_attn lets a query pick a SLICE instead of a SERIES, but the
        # tokens it attends over come from `self.backbone(x)`, which is already
        # globally average-pooled over H and W. So a query still cannot pick a
        # PLACE. At 384px the ConvNeXt map is 12x12, one cell ~= 12.5 mm, and
        # the published OA criterion is a ">=1 cm area of >50%-thickness loss" -
        # i.e. roughly ONE cell, averaged down 144x before anything sees it.
        # band_pool is a partial patch: it collapses rows entirely and keeps
        # three column bands.
        #
        # This pools the feature map to a KxK grid and lets the finding queries
        # attend over (series x slice x cell). Same zero-init gate, so parity
        # with the control is exact at step 0.
        self.cell_k = int(cell_attn)
        if self.cell_k:
            self.cell_route = nn.MultiheadAttention(dim, n_heads, dropout=drop,
                                                    batch_first=True)
            self.cell_norm = nn.LayerNorm(dim)
            self.cell_gate = nn.Parameter(torch.full((1,), float(gate_init)))
            # Cell position is load-bearing, unlike slice position: after
            # laterality canonicalisation the column axis IS medial-lateral on
            # coronal and axial, which is exactly the distinction the weak
            # findings need. Per series TYPE because the axis means something
            # different per plane (anterior-posterior on sagittal) - the same
            # reasoning SeriesEncoder.n_pos_types already uses. std 0.2 matches
            # SeriesEncoder.POS_STD, set there against a measured token norm.
            self.cell_pos = nn.Parameter(
                torch.randn(n_types, self.cell_k ** 2, dim) * SeriesEncoder.POS_STD)
            # ---- through-plane position -------------------------------
            # Measured: the pipeline sorts slices by ImagePositionPatient and
            # data.py REVERSES sagittal stacks for right knees precisely so the
            # index means medial->lateral consistently - and then hands them to
            # an architecture that cannot read it. SeriesEncoder runs with
            # n_pos=0 in every shipped arm (verified: no `series_enc.pos` key in
            # any of the seven), a transformer without positional encoding is
            # permutation-equivariant, and AttnPool is permutation-invariant.
            # Swapping the medial and lateral halves of every sagittal stack
            # moves predictions 0.0048 - indistinguishable from doing the same
            # on coronal where it means nothing (0.0043), 9x smaller than a
            # mirror and 3x smaller than deleting a whole series.
            #
            # On SAGITTAL the slice index IS the medial-lateral axis, which is
            # where the menisci are read. band_pool addressed medial/lateral for
            # coronal and axial only (there it is the image COLUMN); on sagittal
            # the column axis is anterior-posterior, so that axis was never
            # covered. This is the through-plane twin of the pooling defect.
            #
            # BANDS, not a per-index table: --slice-pos used a learned embedding
            # per slice index and measured null, and thirds is what the medial /
            # centre / lateral distinction actually needs. Bands are also
            # slice-count independent, so a checkpoint is not welded to
            # --n-slices the way SeriesEncoder.pos is.
            self.cell_bands = int(cell_slice_bands)
            if self.cell_bands:
                self.cell_zpos = nn.Parameter(
                    torch.randn(n_types, self.cell_bands, dim) * SeriesEncoder.POS_STD)
            else:
                self.cell_zpos = None
        else:
            self.cell_route = self.cell_norm = self.cell_gate = None
            self.cell_pos = None
            self.cell_zpos = None
            self.cell_bands = 0

    def forward(self, x, stype, mask, seq=None, tokens=None, cells=None):
        cel = None
        if self.cell_route is not None and cells is not None:
            # cells: (B, S, Z, K*K, D). Series identity from the same embeddings
            # the series path uses; cell identity from a per-series-type table,
            # because the column axis is medial-lateral on coronal/axial and
            # anterior-posterior on sagittal.
            cel = (cells
                   + self.type_emb(stype)[:, :, None, None, :]
                   + self.cell_pos[stype][:, :, None, :, :])
            if self.cell_zpos is not None:
                Zc = cells.shape[2]
                zb = (torch.arange(Zc, device=cells.device) * self.cell_bands
                      // Zc).clamp(max=self.cell_bands - 1)          # (Z,)
                cel = cel + self.cell_zpos[stype][:, :, zb][:, :, :, None, :]
        tok = None
        if self.flat_route is not None and tokens is not None:
            # tokens: (B, S, Z, D) straight off SeriesEncoder, pre-pool. Series
            # identity is added per slice from the SAME embeddings the series
            # path uses, so a query still knows which series a slice came from.
            tok = tokens + self.type_emb(stype).unsqueeze(2)
            if self.seq_emb is not None and seq is not None:
                tok = tok + self.seq_emb(seq).unsqueeze(2)
        x = x + self.type_emb(stype)
        if self.seq_emb is not None and seq is not None:
            x = x + self.seq_emb(seq)
        if self.study_pool == "mean":
            # scratch/stackdepth arm E: the deliberately simple aggregation --
            # ONE masked mean over series, shared by all twelve findings. No
            # study transformer, no per-finding query routing. This DELETES two
            # measured parts of the shipped edge (the anatomical routing prior,
            # which exists because Synovitis labels are too flat to teach
            # routing, and the finding-family embedding), so it is EXPECTED to
            # cost the routing-dependent columns. It exists to test whether the
            # whole simple configuration works as a configuration, not to win a
            # column on its own.
            m = mask.to(x.dtype).unsqueeze(-1)                    # (B, S, 1)
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-6)    # (B, D)
            emb = self.norm(pooled.unsqueeze(1).expand(
                -1, self.queries.size(0), -1))
            w = (m.squeeze(-1) / m.squeeze(-1).sum(1, keepdim=True)
                 .clamp(min=1e-6))                                # (B, S)
            return emb, w.unsqueeze(1).expand(-1, self.queries.size(0), -1)
        pad = (mask == 0)
        if self.tr is not None:
            x = self.tr(x, src_key_padding_mask=pad)
        x = torch.nan_to_num(x)
        qs = self.queries
        if self.fam_emb is not None:
            # Added to the SCORED queries only. The auxiliary report queries
            # are training-only and have no clinical family; giving them one
            # would put a training-only object into the shared embedding and
            # change the scored path when report_aux is on.
            qs = qs + self.fam_emb(self._fam_idx)
        if self.aux_queries is not None:
            qs = torch.cat([qs, self.aux_queries], 0)
        q = qs.unsqueeze(0).expand(x.size(0), -1, -1)   # (B, n_out+n_aux, D)
        if self.qsa == "pre":
            q = self._query_self_attend(q)
        # A study with zero valid series would make every key masked and produce
        # NaN; keep the first slot alive so the result stays finite.
        pad = pad.clone()
        pad[:, 0] = False
        am = None
        if self.routing_prior_w:
            # (n_out, n_types) indexed by each series' type -> (B, n_out, S),
            # then repeated per head: a float attn_mask ADDS to the logits, so
            # this tilts routing toward the clinically-preferred series types
            # without forbidding anything. Padded keys stay handled by
            # key_padding_mask. w=0 skips entirely - bit-identical control.
            tilt = self._route_prior.to(x.device)[:, stype]      # (n_out, B, S)
            am = (tilt.permute(1, 0, 2) * self.routing_prior_w) \
                .repeat_interleave(self.n_heads, dim=0)
        out, attn = self.route(q, x, x, key_padding_mask=pad,
                               attn_mask=am, need_weights=True)
        out = self.norm(out)
        if tok is not None:
            B, S, Z, D = tok.shape
            # Padded SERIES mask a whole run of Z slice keys. Same first-slot
            # rescue as above: a study with no valid series would otherwise
            # softmax over an all-masked row and return NaN.
            fpad = (mask == 0).unsqueeze(-1).expand(B, S, Z).reshape(B, S * Z)
            fpad = fpad.clone()
            fpad[:, 0] = False
            fo, _ = self.flat_route(q, tok.reshape(B, S * Z, D),
                                    tok.reshape(B, S * Z, D),
                                    key_padding_mask=fpad, need_weights=False)
            fo = self.flat_norm(fo)
            if hasattr(self, "flat_qmask"):
                fo = fo * self.flat_qmask[:fo.shape[1], None]
            out = out + self.flat_gate * fo
        if cel is not None:
            B, S, Z, C, D = cel.shape
            cpad = (mask == 0)[:, :, None, None].expand(B, S, Z, C).reshape(B, S * Z * C)
            cpad = cpad.clone()
            cpad[:, 0] = False
            co, _ = self.cell_route(q, cel.reshape(B, S * Z * C, D),
                                    cel.reshape(B, S * Z * C, D),
                                    key_padding_mask=cpad, need_weights=False)
            out = out + self.cell_gate * self.cell_norm(co)
        if self.qsa == "post":
            out = self._query_self_attend(out)
        return out, attn

    def _query_self_attend(self, q):
        """qsa block on the SCORED rows of q (B, n_out[+n_aux], D); aux rows
        pass through untouched. Pre-norm, residual through qsa_gate."""
        n = self.queries.shape[0]
        qs_, qa_ = q[:, :n], q[:, n:]
        h = self.qsa_norm(qs_)
        o, _ = self.qsa_attn(h, h, h, need_weights=False)
        qs_ = qs_ + self.qsa_gate * o
        return torch.cat([qs_, qa_], 1) if qa_.shape[1] else qs_


# ---------------------------------------------------------------------------
# PF-RelationNet V0 (scratch/pfrel/DESIGN_V0_20260917.md, PREREG.md)
#
# A SINGLE-COLUMN (PF OA) specialist that replaces "global-pooled slice vector
# -> slice transformer -> study transformer -> 12 queries" with an explicit
# patella<->trochlea SURFACE RELATION read off the retained 12x12 spatial map
# of ONE axial fluid-sensitive series.
#
# Two arms, selected by --pfrel / --pfrel-generic and recorded in ck["args"]:
#   "rel"     - six surface queries (3 patellar P1-3, 3 trochlear T1-3), one
#               cross-attention layer per slice, relational soft matching.
#   "generic" - MANDATORY control (spec s11): the six queries and the whole
#               relational block are replaced by ONE generic attention-pooled
#               128-d slice token, parameter-matched with a small MLP. Encoder,
#               input, through-plane operator, head, schedule and loss are
#               IDENTICAL, so Delta_REL = AUC(rel) - AUC(generic) prices the
#               correspondence mechanism and nothing else.
#
# The flag is PARAMETER-BEARING but the two ON modes have nearly identical key
# sets, so - gem_pool/evidence_mil_v2 precedent - the weights can never say
# WHICH mode is on. ck["args"]["pfrel"] is the only contract; src/ckpt_build.py
# resolves it and refuses a disagreement rather than guessing.
# ---------------------------------------------------------------------------
# data.FINDINGS index of the single column --pfrel trains and serves.
# Verified POSITIONALLY at launch by src/train.py (4-char abbreviations
# collide; never parse these by name prefix).
PF_OA_IDX = 6


class PFRelNet(nn.Module):
    """Spec sections 2-8. Consumes the backbone's (N, C, h, w) feature map for
    N = B*Z slices of ONE series and returns one PF-OA logit per study."""

    N_FAM = 3          # queries per family (P1-3 / T1-3)

    def __init__(self, c_in, dim=128, mode="rel", n_heads=4,
                 lambda_pos=0.1, beta=1.0, drop=0.1):
        super().__init__()
        if mode not in ("rel", "generic"):
            raise ValueError(f"pfrel mode={mode!r}")
        self.mode = mode
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        if self.head_dim * n_heads != dim:
            raise ValueError("pfrel dim must divide by n_heads")
        # s2: no global pool. 1x1 conv 768 -> 128 over the retained map.
        self.proj = nn.Conv2d(c_in, dim, 1)
        # coordinate embedding on (x, y, z), normalised to [-1, 1].
        self.coord = nn.Sequential(nn.Linear(3, dim), nn.GELU(),
                                   nn.Linear(dim, dim))
        self.tok_norm = nn.LayerNorm(dim)
        if mode == "rel":
            # s3: six global trainable surface queries, ONE cross-attention
            # layer per slice (Q 6x128, K/V 144x128, 4 heads x 32).
            self.q = nn.Parameter(torch.randn(2 * self.N_FAM, dim) * 0.02)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.q_proj = nn.Linear(dim, dim)
            self.o_proj = nn.Linear(dim, dim)
            # s4: weak additive positional bias lambda * B(x, y). B is FIXED and
            # weak (patella mildly anterior, trochlea central/posterior); only
            # the scale is learnable, initialised at 0.1 of the attention-logit
            # scale. NO fixed ROIs, masks, segmenter outputs or facets - those
            # are the closed compartment-space forms (anatroi, structruth).
            self.lam = nn.Parameter(torch.full((2 * self.N_FAM,),
                                               float(lambda_pos)))
            # s5: relational matching projection and the distance penalty.
            self.W = nn.Linear(dim, dim, bias=False)
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            # s11 control: one generic gated attention pool over the SAME 144
            # tokens, no queries, no positional prior, no correspondence.
            self.gpool = AttnPool(dim)
            # parameter-matching MLP (stands in for the query set + the
            # cross-attention projections the rel arm carries).
            self.gmlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                      nn.Linear(2 * dim, dim))
        # s5 tail / s6: the relation MLP is SHARED by both arms (same shape,
        # same init path), so the arms differ only in what fills its input.
        self.rel_mlp = nn.Sequential(nn.Linear(4 * dim + 3, 2 * dim), nn.GELU(),
                                     nn.Linear(2 * dim, dim))
        self.rel_norm = nn.LayerNorm(dim)
        self.pool_w = nn.Linear(dim, 1)        # s6: a_ij = w^T R_ij
        # s7: two residual depthwise-Conv1D blocks over the slice axis.
        self.tp = nn.ModuleList([_PFRelTPBlock(dim, k) for k in (3, 5)])
        self.zpos = nn.Linear(1, dim)
        # s8: evidence head.
        self.drop = nn.Dropout(drop)
        self.w_s = nn.Linear(dim, 1)
        self.w_u = nn.Linear(dim, 1)
        self.tau = nn.Parameter(torch.zeros(()))
        self.kappa = nn.Parameter(torch.tensor(1.0))
        self.out = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        # train-time auxiliaries (s9), stashed for the trainer.
        self._aux = None

    # -- fixed weak prior ---------------------------------------------------
    def _bias(self, yy):
        """B(x, y) per query family on an AXIAL slice.

        After preprocess.py's canonicalisation the row axis of an axial slice
        runs anterior -> posterior, so anteriority is simply -y. The patellar
        family gets +(-y) and the trochlear family gets -(-y) (central/
        posterior-to-patella). Magnitude is bounded by |y| <= 1 and scaled by
        the learnable lambda, i.e. this is a BIAS, never a mask.
        """
        ant = -yy                                     # (T,)
        b = torch.stack([ant] * self.N_FAM + [-ant] * self.N_FAM, 0)  # (6,T)
        return b

    def forward(self, fm, B, Z):
        """fm: (B*Z, C, h, w) feature map of the chosen axial series."""
        N, C, h, w = fm.shape
        if N != B * Z:
            raise ValueError(f"pfrel: got {N} maps for B={B} Z={Z}")
        x = self.proj(fm).flatten(2).transpose(1, 2)          # (N, T, D)
        T = h * w
        dev, dt = x.device, x.dtype
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=dev, dtype=dt),
            torch.linspace(-1, 1, w, device=dev, dtype=dt), indexing="ij")
        xs, ys = xs.reshape(-1), ys.reshape(-1)               # (T,)
        zs = torch.linspace(-1, 1, Z, device=dev, dtype=dt)   # (Z,)
        zt = zs.repeat(B).view(N, 1, 1).expand(N, T, 1)
        coord = torch.cat([xs.view(1, T, 1).expand(N, T, 1),
                           ys.view(1, T, 1).expand(N, T, 1), zt], -1)
        x = self.tok_norm(x + self.coord(coord))
        xy = torch.stack([xs, ys], -1)                        # (T, 2)

        aux = {}
        if self.mode == "rel":
            R, cent = self._relational(x, xy, N, T)           # (N,3,D),(N,6,2)
            aux = self._aux_terms(cent, B, Z)
        else:
            s, _ = self.gpool(x)                              # (N, D)
            s = s + self.gmlp(s)
            z = torch.cat([s, s, torch.zeros_like(s), s * s,
                           x.new_zeros(N, 3)], -1)
            R = self.rel_norm(self.rel_mlp(z)).unsqueeze(1)   # (N, 1, D)
        # s6: within-slice pooling AFTER the relations exist.
        a = torch.softmax(self.pool_w(R), dim=1)              # (N, J, 1)
        r = (a * R).sum(1).view(B, Z, self.dim)               # (B, Z, D)
        # s7
        r = r + self.zpos(zs.view(1, Z, 1).expand(B, Z, 1))
        for blk in self.tp:
            r = blk(r)
        # s8
        r = self.drop(r)
        sev = self.w_s(r).squeeze(-1)                         # (B, Z)
        ext = self.w_u(r).squeeze(-1)
        v = torch.stack([
            sev.mean(1),
            torch.quantile(sev.float(), 0.80, dim=1).to(sev.dtype),
            torch.quantile(sev.float(), 0.95, dim=1).to(sev.dtype),
            sev.max(1).values,
            torch.logsumexp(sev, dim=1) - math.log(Z),
            torch.sigmoid(self.kappa * (ext - self.tau)).mean(1),
        ], -1)                                                # (B, 6)
        self._aux = aux
        return self.out(v).squeeze(-1)                        # (B,)

    # -- s3 + s5 ------------------------------------------------------------
    def _relational(self, x, xy, N, T):
        H, dh = self.n_heads, self.head_dim
        q = self.q_proj(self.q).view(1, 2 * self.N_FAM, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(N, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(N, T, H, dh).transpose(1, 2)
        att = (q * (dh ** -0.5)) @ k.transpose(-1, -2)        # (N,H,6,T)
        att = att + (self.lam.view(1, 1, -1, 1)
                     * self._bias(xy[:, 1]).view(1, 1, 2 * self.N_FAM, T))
        att = torch.softmax(att.float(), -1).to(x.dtype)
        o = self.o_proj((att @ v).transpose(1, 2).reshape(N, 2 * self.N_FAM,
                                                          self.dim))
        # attention centroids c_ij, averaged over heads (s3).
        cent = att.mean(1) @ xy                               # (N, 6, 2)
        p, t = o[:, :self.N_FAM], o[:, self.N_FAM:]
        cp, ct = cent[:, :self.N_FAM], cent[:, self.N_FAM:]
        m = (self.W(p) @ self.W(t).transpose(1, 2)) * (self.dim ** -0.5)
        d2 = ((cp.unsqueeze(2) - ct.unsqueeze(1)) ** 2).sum(-1)   # (N,3,3)
        alpha = torch.softmax((m - self.beta.abs() * d2).float(), -1).to(x.dtype)
        tt = alpha @ t                                        # (N, 3, D)
        dc = (alpha @ ct) - cp                                # (N, 3, 2)
        z = torch.cat([p, tt, (p - tt).abs(), p * tt, dc,
                       dc.norm(dim=-1, keepdim=True)], -1)
        return self.rel_norm(self.rel_mlp(z)), cent

    # -- s9 -----------------------------------------------------------------
    def _aux_terms(self, cent, B, Z, margin=0.4):
        c = cent.view(B, Z, 2 * self.N_FAM, 2)
        smooth = (c[:, 1:] - c[:, :-1]).abs().mean()
        f = [c[:, :, :self.N_FAM], c[:, :, self.N_FAM:]]
        hinge = []
        for cf in f:
            d = (cf.unsqueeze(3) - cf.unsqueeze(2)).norm(dim=-1)   # (B,Z,3,3)
            iu = torch.triu(torch.ones_like(d), diagonal=1) > 0
            hinge.append(torch.relu(margin - d)[iu].mean())
        return {"smooth": smooth, "collapse": torch.stack(hinge).mean()}


# ---------------------------------------------------------------------------
# Bone-WSRPN V0 (scratch/bonewsrpn/CANDIDATE.md, PREREG.md)
#
# A TWO-COLUMN (Contusion, Fracture) osseous-lesion specialist. It reads ONE
# sagittal fluid-sensitive and ONE coronal fluid-sensitive series, keeps the
# 24x24 and 12x12 spatial maps (two-level FPN -> 128-d), mixes them locally in
# 3D, and then either
#   "roi"   - proposes K=4 differentiable soft 3D regions per series (WSRPN
#             style: query cross-attention -> centroid + softplus scale ->
#             sigmoid-product soft mask), pools a CORE and a SHELL vector from
#             each, and runs smooth MIL over the union of 2*K objects, or
#   "dense" - the MANDATORY control: identical encoder, FPN, 3D blocks, heads,
#             schedule and loss, with the proposals removed; per-VOXEL logits
#             on the same feature volume, smooth MIL over the voxels,
#             parameter-matched with a small MLP.
#
# Both arms end in the SAME production osseous factorisation (osseous_factorize,
# p_C = p_E (1 - p_L), p_F = p_E p_L), so Delta_BONE = AUC(roi) - AUC(dense)
# prices the PROPOSALS and nothing else. Per-object heads are e_k (osseous
# lesion evidence) and l_k (cortical line | lesion); the study-level pair is
#   z_E = LSE_k(e_k)              z_L = LSE_k(e_k + l_k) - LSE_k(e_k)
# i.e. the line logit is the evidence-weighted soft average of the per-object
# line logits - the conditional the factorisation needs. This REPLACES the
# design's independent c_k/f_k pair (which cannot feed p_E/p_L) and is the one
# declared deviation from CANDIDATE.md's head layout.
#
# PARAMETER-BEARING, but the two ON modes share most key names, so - pfrel /
# gem_pool / evidence_mil_v2 precedent - ck["args"]["bone"] is the ONLY
# contract; src/ckpt_build.py resolves it and refuses a disagreement.
CONT_IDX = 10
FRAC_IDX = 11


class _Bone3DBlock(nn.Module):
    """Residual depthwise Conv3D 3x3x3 + pointwise, on (N, D, Z, H, W)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.pw1 = nn.Conv3d(dim, 2 * dim, 1)
        self.pw2 = nn.Conv3d(2 * dim, dim, 1)

    def forward(self, x):
        h = self.dw(self.norm(x))
        return x + self.pw2(F.gelu(self.pw1(F.gelu(h))))


class BoneWSRPN(nn.Module):
    """CANDIDATE.md design. Consumes the backbone's stage-3 (N,C3,2h,2w) and
    stage-4 (N,C4,h,w) maps for N = B*S*Z slices of S <= 2 series and returns
    (z_E, z_L) per study."""

    def __init__(self, c3, c4, dim=128, mode="roi", k=4, n_heads=4, tau=0.1,
                 shell=2.0, n_stype=6, drop=0.1):
        super().__init__()
        if mode not in ("roi", "dense"):
            raise ValueError(f"bone mode={mode!r}")
        self.mode, self.dim, self.k = mode, dim, int(k)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        if self.head_dim * n_heads != dim:
            raise ValueError("bone dim must divide by n_heads")
        self.tau = float(tau)
        self.shell = float(shell)
        # two-level FPN: 12x12x768 -> 128, upsampled onto 24x24x384 -> 128.
        self.p4 = nn.Conv2d(c4, dim, 1)
        self.p3 = nn.Conv2d(c3, dim, 1)
        self.fuse = nn.Conv2d(dim, dim, 3, padding=1)
        # two residual depthwise Conv3D blocks on the FUSED P3 volume.
        self.tp = nn.ModuleList([_Bone3DBlock(dim) for _ in range(2)])
        self.coord = nn.Sequential(nn.Linear(3, dim), nn.GELU(),
                                   nn.Linear(dim, dim))
        self.tok_norm = nn.LayerNorm(dim)
        if mode == "roi":
            self.q = nn.Parameter(torch.randn(self.k, dim) * 0.02)
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.o_proj = nn.Linear(dim, dim)
            self.scale_head = nn.Linear(dim, 3)
            self.tok = nn.Sequential(nn.Linear(3 * dim + 7, 2 * dim), nn.GELU(),
                                     nn.Linear(2 * dim, dim))
        else:
            # control: no queries, no masks, no pooling geometry. The MLP
            # stands in for the query set + the four attention projections.
            self.dmlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                      nn.Linear(2 * dim, dim))
            self.tok = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                     nn.Linear(2 * dim, dim))
        self.stype_emb = nn.Embedding(n_stype, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.h_e = nn.Linear(dim, 1)     # osseous lesion evidence
        self.h_l = nn.Linear(dim, 1)     # cortical line | lesion
        # MIL temperature, learnable, shared by both arms. LSE is length-
        # normalised (log-mean-exp) so the 8-object and the 13,824-voxel arms
        # live on the same scale.
        self.mil_t = nn.Parameter(torch.tensor(0.5))
        self._aux = None                 # diversity term, stashed for train.py
        self._zE = None                  # z_E, stashed for the 0.2 L_bone term

    # -- soft sigmoid-product 3D mask ---------------------------------------
    def _mask(self, coords, c, s):
        """coords (M,3) in [-1,1]; c (N,K,3); s (N,K,3) -> (N,K,M)."""
        d = (coords.view(1, 1, -1, 3) - c.unsqueeze(2)).abs()
        u = torch.sigmoid((1.0 - d / s.unsqueeze(2).clamp(min=1e-3)) / self.tau)
        return u.prod(-1)

    def forward(self, fm3, fm4, B, S, Z, stype, mask):
        N, C4, h, w = fm4.shape
        if N != B * S * Z:
            raise ValueError(f"bone: got {N} maps for B={B} S={S} Z={Z}")
        p = self.fuse(F.interpolate(self.p4(fm4), size=fm3.shape[-2:],
                                    mode="nearest") + self.p3(fm3))
        H, W = p.shape[-2:]
        v = p.view(B * S, Z, self.dim, H, W).permute(0, 2, 1, 3, 4)
        for blk in self.tp:
            v = blk(v)                                     # (BS, D, Z, H, W)
        BS = B * S
        x = v.flatten(2).transpose(1, 2)                   # (BS, M, D)
        M = x.shape[1]
        dev, dt = x.device, x.dtype
        zs, ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, Z, device=dev, dtype=dt),
            torch.linspace(-1, 1, H, device=dev, dtype=dt),
            torch.linspace(-1, 1, W, device=dev, dtype=dt), indexing="ij")
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], -1)
        x = self.tok_norm(x + self.coord(coords).unsqueeze(0))
        se = self.stype_emb(stype.reshape(BS)).unsqueeze(1)     # (BS,1,D)

        aux = {}
        if self.mode == "roi":
            q = self.q_proj(self.q).view(self.k, self.n_heads, self.head_dim)
            kk = self.k_proj(x).view(BS, M, self.n_heads, self.head_dim)
            vv = self.v_proj(x).view(BS, M, self.n_heads, self.head_dim)
            att = torch.einsum("khd,bmhd->bhkm", q, kk) * (self.head_dim ** -0.5)
            att = torch.softmax(att.float(), -1).to(dt)         # (BS,h,K,M)
            a = self.o_proj(torch.einsum("bhkm,bmhd->bkhd", att, vv)
                            .reshape(BS, self.k, self.dim))     # (BS,K,D)
            am = att.mean(1)                                    # (BS,K,M)
            c = am @ coords                                     # (BS,K,3)
            s = F.softplus(self.scale_head(a)) + 0.05           # (BS,K,3)
            mc = self._mask(coords, c, s)                       # core
            ms = (self._mask(coords, c, self.shell * s) - mc).clamp(min=0)
            gc = (mc @ x) / mc.sum(-1, keepdim=True).clamp(min=1e-4)
            gs = (ms @ x) / ms.sum(-1, keepdim=True).clamp(min=1e-4)
            occ = mc.mean(-1, keepdim=True)                      # (BS,K,1)
            t = self.tok(torch.cat([gc, gs, (gc - gs).abs(), c, s, occ], -1))
            t = self.norm(t + se)                                # (BS,K,D)
            aux = {"mc": mc}
        else:
            t = self.norm(self.tok(x + self.dmlp(x)) + se)       # (BS,M,D)
        t = self.drop(t)
        e = self.h_e(t).squeeze(-1).float()                      # (BS,J)
        l = self.h_l(t).squeeze(-1).float()
        J = e.shape[1]
        # series validity -> object validity. A study with no usable series is
        # all-invalid and gets an exact zero logit pair (serve-time fallback).
        mv = mask.reshape(BS, 1).expand(BS, J).reshape(B, S * J).float()
        e = e.reshape(B, S * J)
        l = l.reshape(B, S * J)
        tmil = self.mil_t.abs().clamp(min=0.05).float()
        nv = mv.sum(1).clamp(min=1.0)
        neg = torch.full_like(e, -1e4)
        ev = torch.where(mv > 0, e, neg)
        zE = tmil * (torch.logsumexp(ev / tmil, 1) - nv.log())
        zEL = tmil * (torch.logsumexp(torch.where(mv > 0, e + l, neg) / tmil, 1)
                      - nv.log())
        zL = zEL - zE
        any_v = (mv.sum(1) > 0).float()
        zE, zL = zE * any_v, zL * any_v
        if self.mode == "roi":
            # diversity: pairwise soft-mask overlap, counted ONLY where at
            # least two objects of a series are actually "on"
            # (sigmoid(e) > 0.5). The control has no masks and adds nothing.
            mc = aux["mc"]
            on = (torch.sigmoid(e.reshape(B, S, J).detach()) > 0.5).float() \
                .reshape(BS, J)
            gate = (on.sum(1) >= 2).float()                      # (BS,)
            mass = mc.sum(-1).clamp(min=1e-4)                    # (BS,K)
            inter = torch.einsum("bim,bjm->bij", mc, mc)
            denom = torch.minimum(mass.unsqueeze(2), mass.unsqueeze(1))
            iou = inter / denom.clamp(min=1e-4)
            pair = on.unsqueeze(2) * on.unsqueeze(1)
            eye = torch.eye(J, device=dev).view(1, J, J)
            pair = pair * (1 - eye)
            div = ((iou * pair).sum((1, 2)) * gate).sum() \
                / (pair.sum((1, 2)) * gate).sum().clamp(min=1.0)
            self._aux = {"div": div}
        else:
            self._aux = None
        self._zE = zE
        return zE, zL


class _PFRelTPBlock(nn.Module):
    """s7: LN -> depthwise Conv1D(k) -> GELU -> pointwise D->2D->D, residual."""

    def __init__(self, dim, k):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim)
        self.pw = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                nn.Linear(2 * dim, dim))

    def forward(self, x):                                  # (B, Z, D)
        h = self.norm(x)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        return x + self.pw(F.gelu(h))


class EvAdapter(nn.Module):
    """--ev-adapter (scratch/evform/PREREG.md): conditional evidence FORMATION.

    A residual, soft-routed low-rank adapter on the 768-d contextualised slice
    token h, inserted immediately BEFORE the per-slice evidence readout:

        h~ = h + sum_k pi_k(h) . U_k phi(V_k LN(h)),   pi = softmax(W_r LN(h))

    mode="moe": K experts (default 4) of bottleneck width W (default 32).
    mode="mlp": ONE expert, NO router - the matched-CAPACITY control. Its width
        is DERIVED from (K, W) so the two forms have the same parameter count
        to <1%; --ev-adapter-width is always the per-expert moe bottleneck.

    Init is FUNCTION-PRESERVING but not doubly zero-gated (collaborator reply
    12, notes/COLLAB_REPLY12_20260922.md): U_k = 0 so h~ == h at step 0, while
    V_k and W_route get ORDINARY normal init - the routing is therefore
    non-uniform and slice-varying from step 0, instead of the zero-gate-on-
    zero-gate that produced the dead gates in scratch/spatev3.
    Structural note (asserted in scratch/evform/test_evform.py): with U = 0
    every expert output is exactly zero at step 0, so dL/dpi = 0 there for ANY
    function-preserving init; V and W_route carry gradient from step 1 onward,
    as soon as U is off zero.

    One LayerNorm is shared by the experts and the router. The collaborator's
    sketch applies the experts to the raw h, but the slice transformer is
    norm_first and therefore returns UNNORMALISED tokens - ev_head normalises
    for exactly that reason - so an unnormalised V_k h would be initialised at
    an arbitrary scale. Documented deviation; it cannot break the
    function-preserving property (U = 0).
    """

    @staticmethod
    def mlp_width(dim, k, width):
        """Bottleneck width whose single-expert parameter count matches the
        k-expert moe (U+V = 2*dim*w vs k*2*dim*width + router k*(dim+1))."""
        return int(round(k * width + k * (dim + 1) / (2.0 * dim)))

    def __init__(self, dim, mode, k=4, width=32):
        super().__init__()
        if mode not in ("mlp", "moe"):
            raise ValueError(f"EvAdapter mode={mode!r}")
        self.mode = str(mode)
        self.n_exp = int(k) if mode == "moe" else 1
        self.width = int(width) if mode == "moe" else self.mlp_width(
            dim, int(k), int(width))
        self.norm = nn.LayerNorm(dim)
        self.V = nn.Parameter(torch.empty(self.n_exp, dim, self.width))
        # U = 0 -> the adapter is the identity at init (bitwise: x + 0.0).
        self.U = nn.Parameter(torch.zeros(self.n_exp, self.width, dim))
        nn.init.normal_(self.V, std=dim ** -0.5)
        self.route = None
        if mode == "moe":
            self.route = nn.Linear(dim, self.n_exp)
            nn.init.normal_(self.route.weight, std=dim ** -0.5)
            nn.init.zeros_(self.route.bias)
        self.last_pi = None

    def forward(self, h):
        """h (..., D) -> (..., D), same dtype/shape."""
        sh = h.shape
        x = h.reshape(-1, sh[-1])
        xn = self.norm(x)
        y = torch.einsum("nd,kdw->nkw", xn, self.V)
        y = F.gelu(y)
        y = torch.einsum("nkw,kwd->nkd", y, self.U)
        if self.route is not None:
            pi = torch.softmax(self.route(xn).float(), dim=-1)
            self.last_pi = pi.detach()
            y = (pi.to(y.dtype).unsqueeze(-1) * y).sum(1)
        else:
            self.last_pi = None
            y = y.sum(1)
        return (x + y).reshape(sh)


class _SeqAdaptStage(nn.Module):
    """scratch/seqadapt (--seq-adapt R): per-SEQUENCE-TYPE conditioning of ONE
    backbone stage output (N, C, H, W), applied by a forward hook so the
    backbone's own state-dict keys are untouched:
        y = x * (1 + gamma[t]) + beta[t] + up[t] @ gelu(down[t] @ x)
    with t = the row's series type (6 = 3 planes x FS/non-FS). Identity at
    init (gamma = beta = up = 0, so y == x exactly) with a LIVE gradient path:
    gamma/beta/up receive nonzero gradients on step 1 (down is random), and
    down becomes live once up moves (no zero-init scalar gate; cf. the stage-3
    adapter collapse). down is drawn from a PRIVATE generator so the global
    RNG stream - hence every other parameter's init - equals the flag-off
    run's. Rows are supplied per forward through the shared ctx dict."""
    def __init__(self, c, n_types, rank, ctx, seed):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.gamma = nn.Parameter(torch.zeros(n_types, c))
        self.beta = nn.Parameter(torch.zeros(n_types, c))
        self.down = nn.Parameter(torch.randn(n_types, rank, c, generator=g) * c ** -0.5)
        self.up = nn.Parameter(torch.zeros(n_types, c, rank))
        self.c = int(c)
        self._ctx = ctx

    def hook(self, mod, inp, out):
        rows = self._ctx.get("rows")
        if rows is None or out.ndim != 4 or out.shape[0] != rows.shape[0] \
                or out.shape[1] != self.c:
            raise RuntimeError(
                f"--seq-adapt hook: stage output {tuple(out.shape)} vs rows "
                f"{None if rows is None else tuple(rows.shape)} / C={self.c}; "
                "refusing to condition on misaligned series types.")
        n, c, h, w = out.shape
        dt = out.dtype
        y = out * (1 + self.gamma[rows].to(dt))[:, :, None, None] \
            + self.beta[rows].to(dt)[:, :, None, None]
        z = torch.bmm(self.down[rows].to(dt), out.reshape(n, c, h * w))   # (N, r, HW)
        z = torch.bmm(self.up[rows].to(dt), F.gelu(z))                     # (N, C, HW)
        return y + z.reshape(n, c, h, w)


class KneeNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=True, n_out=12,
                 drop=0.1, slice_ctx=3, img_norm="imagenet", stack_depth=False, study_pool="route", text_dim=0, n_seq=0, n_report_aux=0,
                 n_slice_pos=0, slice_pos_types=0, band_pool=False, band_delta=0.125,
                 routing_prior=0.0, slice_enc="tr", flat_attn=False, cell_attn=0,
                 cell_stage=-1, gate_init=0.1, cell_slice_bands=3, gem_p=1.0,
                 flat_queries=None,
                 osseous_factor=False, oa_gate=False, query_family="none",
                 moaks_blocks=(), aux_seg_classes=0, slice3d="none",
                 blurpool=0, qsa="none", qsa_gate_init=0.1, drop_path=0.0,
                 evidence_mil=False, evidence_mil_v2="off",
                 evdrop=False, evdrop_p=0.5,
                 pat=False, pat_grid=2,
                 darc_local=False,
                 ev_adapter="off", ev_adapter_k=4, ev_adapter_width=32,
                 evmil_prior="clinical",
                 evmil_tau_init=1.0, evmil_dif_init="mean",
                 ev2_gamma_init=0.01, bypass_study_tr=False,
                 syno_specialist=False, syno_gamma_init=0.01,
                 pfrel="off", pfrel_lambda_pos=0.1, pfrel_beta=1.0,
                 bone="off", bone_k=4, bone_tau=0.1, bone_shell=2.0,
                 rubric_pool=False, xfuse=False, ps25=False,
                 bn_valid_only=False, seq_adapt=0, tfcrop_types=0):
        super().__init__()
        # scratch/truebatch/PREREG.md: --bn-valid-only. The backbone runs ONLY
        # on the slices of real (mask==1) series; padded series slots get a
        # zero feature row. Needed for a BatchNorm trunk in TRAIN mode, where
        # the constant padded images would otherwise enter the batch
        # statistics. False (default) leaves the forward byte-identical. Only
        # wired on the plain gem_p==1 path; every other branch refuses below.
        self.bn_valid_only = bool(bn_valid_only)
        # scratch/divarch PS25: per-series 2.5D LATE FUSION. Each series is scored
        # alone from mean+max pooled slice features (+ a sequence-type embedding)
        # by a shared per-series 12-logit head; study logits = masked MEAN of the
        # series logits. Slice transformer / study transformer are bypassed
        # (still constructed, unused). args is the contract (ckpt_build).
        self.ps25 = bool(ps25)
        # --pfrel / --pfrel-generic (scratch/pfrel/PREREG.md). "off" (default)
        # constructs NOTHING below and no forward line runs, so every
        # pre-existing arm is byte-identical. The two ON modes ("rel",
        # "generic") share almost every key name, so the state dict can never
        # say which is on: ck["args"]["pfrel"] is the ONLY contract
        # (gem_pool / evidence_mil_v2 precedent) and ckpt_build.py refuses a
        # disagreement rather than guessing.
        self.pfrel = str(pfrel or "off")
        if self.pfrel not in ("off", "rel", "generic"):
            raise ValueError(f"pfrel={self.pfrel!r}")
        # --- Bone-WSRPN V0 (scratch/bonewsrpn/PREREG.md) -------------------
        # "off" (default) is the pre-existing path byte-for-byte: nothing is
        # constructed and no forward line runs.
        self.bone = str(bone or "off")
        if self.bone not in ("off", "roi", "dense"):
            raise ValueError(f"bone={self.bone!r}")
        if self.bone != "off" and self.pfrel != "off":
            raise ValueError("--bonewsrpn and --pfrel are different specialist "
                             "specs and cannot both be on")
        self.bone_k = int(bone_k)
        self.bone_tau = float(bone_tau)
        self.bone_shell = float(bone_shell)
        if self.bone != "off":
            _badb = [n for n, v in (("evidence_mil", bool(evidence_mil)),
                                    ("stack_depth", bool(stack_depth)),
                                    ("band_pool", bool(band_pool)),
                                    ("cell_attn", bool(cell_attn)),
                                    ("flat_attn", bool(flat_attn)),
                                    ("slice3d", slice3d not in (None, "none")),
                                    ("aux_seg_classes", bool(aux_seg_classes)),
                                    ("text_dim", bool(text_dim)),
                                    ("n_report_aux", bool(n_report_aux)),
                                    ("moaks_blocks", bool(moaks_blocks)),
                                    ("osseous_factor", bool(osseous_factor)),
                                    ) if v]
            if _badb:
                raise ValueError(
                    "--bonewsrpn replaces the whole slice/study aggregation "
                    "path with the osseous lesion-object specialist (and "
                    "applies osseous_factorize itself) and is incompatible "
                    "with: " + ", ".join(_badb))
        if self.pfrel != "off":
            _bad = [n for n, v in (("evidence_mil", bool(evidence_mil)),
                                   ("stack_depth", bool(stack_depth)),
                                   ("band_pool", bool(band_pool)),
                                   ("cell_attn", bool(cell_attn)),
                                   ("flat_attn", bool(flat_attn)),
                                   ("slice3d", slice3d not in (None, "none")),
                                   ("aux_seg_classes", bool(aux_seg_classes)),
                                   ("text_dim", bool(text_dim)),
                                   ("n_report_aux", bool(n_report_aux)),
                                   ("moaks_blocks", bool(moaks_blocks)),
                                   ) if v]
            if _bad:
                raise ValueError(
                    "--pfrel replaces the whole slice/study aggregation path "
                    "with the PF-OA relational specialist and is incompatible "
                    "with: " + ", ".join(_bad))
        if bypass_study_tr and evidence_mil:
            raise ValueError("--bypass-study-transformer is the attribution "
                             "CONTROL for --evidence-mil; the two cannot be "
                             "on together (evidence-mil has no study_enc).")
        # --evidence-mil (scratch/evmil/PREREG.md): pathology-specific evidence
        # accumulation REPLACES "attn-pooled series token -> study transformer
        # -> 12 cross-attention queries". When on, study_enc and its queries
        # are NOT CONSTRUCTED (no dead parameters) and the per-series aux head
        # is replaced by the per-series pooled evidence itself, which is the
        # serving path rather than a train-only head. Off by default and the
        # off path is byte-identical: nothing below this flag runs and no
        # module/ buffer is added. Stored in ck["args"]; every consumer that
        # rebuilds a model from a checkpoint MUST read args["evidence_mil"]
        # (the weights do carry an unambiguous signature - ev_R / ev_head.* -
        # but args is the contract).
        self.evidence_mil = bool(evidence_mil)
        # --tfcrop-root (scratch/tfcrop_20261011): number of EXTRA series-type
        # ids (crop series) after the 6 standard ones. 0 -> nothing changes.
        # Wired ONLY through the evidence-MIL head (ev_R gains columns);
        # refuse every other type-indexed module rather than clamp/skip.
        self.tfcrop_types = int(tfcrop_types)
        if self.tfcrop_types:
            if not self.evidence_mil:
                raise NotImplementedError("--tfcrop-root requires --evidence-mil (StudyEncoder type_emb not extended)")
            if slice_pos_types or seq_adapt or routing_prior:
                raise NotImplementedError("--tfcrop-root is not wired with slice_pos_types/seq_adapt/routing_prior")
        # --evdrop (scratch/evdrop/PREREG.md): TRAIN-ONLY dominant-evidence
        # dropout. In TRAIN mode only, and only on a fraction evdrop_p of
        # forwards, a SECOND per-series pool is computed in which the single
        # top-evidence slice of every (study, series, FOCAL finding) triple is
        # removed before the LSE; the trainer adds a focal-only loss on the
        # resulting study logits. No parameter, no buffer, no state-dict key
        # and no eval-mode effect exist: `self.training` gates the whole
        # branch, so the serve forward is byte-identical and no kernel port is
        # required (src/ckpt_build.py:TRAIN_ONLY_FLAGS).
        self.evdrop = bool(evdrop)
        self.evdrop_p = float(evdrop_p)
        # Transient stash for the masked-view logits, read by src/train.py
        # after each forward. A plain attribute (never a buffer) so it cannot
        # reach the state dict; reset to None at the top of every evidence
        # forward and again by the trainer after it is consumed.
        self._evdrop_logits = None
        if self.evdrop:
            # Refuse-not-skip: every one of these would silently change WHAT
            # is masked or WHAT the masked view means.
            if not evidence_mil:
                raise ValueError("evdrop needs evidence_mil: the masking acts "
                                 "on the per-slice evidence tensor, which "
                                 "only the evidence-MIL head builds")
            if str(evidence_mil_v2 or "off") != "off":
                raise ValueError("evdrop with evidence_mil_v2 is refused: the "
                                 "v2 residual is added to ep AFTER the LSE, so "
                                 "the masked view would keep the residual of "
                                 "the slice it just removed")
            if syno_specialist:
                raise ValueError("evdrop with syno_specialist is refused: the "
                                 "Synovitis residual is a study-level object "
                                 "and Synovitis must stay exactly untouched")
            if not (0.0 < self.evdrop_p <= 1.0):
                raise ValueError(f"evdrop_p must be in (0, 1], got "
                                 f"{self.evdrop_p!r}")
        # --pat (scratch/shortcutarms/PREREG.md §PAT; src/pat.py): TRAIN-ONLY
        # patching-based counterfactual training. On every TRAINING forward the
        # slice features are re-encoded per SLAB (pat_grid contiguous slot
        # ranges of one series), read by the same ev_head and pooled by the
        # same V1 pool; the pivot-softmax aggregate of those patch logits is
        # stashed for the trainer. No parameter, buffer or state-dict key and
        # no eval-mode effect: the serve forward is byte-identical and no
        # kernel port is required (ckpt_build.TRAIN_ONLY_FLAGS).
        self.pat = bool(pat)
        self.pat_grid = int(pat_grid)
        self._pat_logits = None
        if self.pat:
            _pb = [n for n, v in (
                ("evidence_mil off", not evidence_mil),
                ("evidence_mil_v2", str(evidence_mil_v2 or "off") != "off"),
                ("syno_specialist", bool(syno_specialist)),
                ("ev_adapter", str(ev_adapter or "off") != "off"),
                ("evdrop", bool(evdrop)),
                ("n_slice_pos", int(n_slice_pos or 0) > 0),
                ("slice_enc", str(slice_enc) != "tr")) if v]
            if _pb:
                raise ValueError(f"pat is refused with {_pb}: the patch view "
                                 f"replicates exactly the V1 evidence pool on a "
                                 f"permutation-invariant slice transformer")
            if self.pat_grid < 2:
                raise ValueError(f"pat_grid must be >= 2, got {pat_grid!r}")
        # --darc-local (scratch/shortcutarms/PREREG.md §3, src/darc.py; HELD):
        # per-finding total-direct-effect evidence, every companion readout
        # direction projected out of W_k. SERVE-PATH, zero parameters; the
        # Synovitis row is kept verbatim via a NON-persistent buffer (state
        # dict unchanged). ckpt_build: model-affecting flag, args-only contract.
        self.darc_local = bool(darc_local)
        if self.darc_local:
            if not evidence_mil:
                raise ValueError("darc_local needs evidence_mil")
            if self.pat:
                raise ValueError("darc_local with pat is refused: the patch "
                                 "view reads ev_head directly")
            from data import FINDINGS as _FD
            self.register_buffer(
                "darc_keep",
                torch.tensor([f == "Synovitis" for f in _FD]),
                persistent=False)
        # --ev-adapter (scratch/evform/PREREG.md): conditional evidence
        # FORMATION. A residual, soft-routed low-rank adapter sits immediately
        # BEFORE the per-slice evidence readout (ev_head = LN + dropout +
        # Linear), i.e. on the 768-d contextualised slice token h:
        #     h~ = h + sum_k pi_k(h) U_k phi(V_k LN(h)),  pi = softmax(W_r LN(h))
        # "off" (default) constructs NOTHING and no forward line runs, so every
        # pre-existing arm is byte-identical. "mlp" is the matched-CAPACITY
        # control (one expert, no router, width derived to match the moe
        # parameter count). The adapter is PARAMETER-BEARING (ev_adapter.*
        # keys), so strict=True would raise on a mis-served checkpoint anyway -
        # but "the load happened to fail" is not a contract: ck["args"]
        # ["ev_adapter"] is the contract (src/ckpt_build.py cross-checks both
        # directions) and the kernel refuses it until a port exists.
        self.ev_adapter = str(ev_adapter or "off")
        self.ev_adapter_k = int(ev_adapter_k)
        self.ev_adapter_width = int(ev_adapter_width)
        # Validated refuse-not-skip even when the adapter is off, so a typo can
        # never sit silently in ck["args"] (the evmil_prior precedent).
        if self.ev_adapter not in ("off", "mlp", "moe"):
            raise ValueError(f"ev_adapter={self.ev_adapter!r} is not one of "
                             f"off|mlp|moe")
        if self.ev_adapter != "off":
            if not evidence_mil:
                raise ValueError("ev_adapter needs evidence_mil: it acts on the "
                                 "slice token the EV evidence readout consumes")
            if self.ev_adapter_k < 1 or self.ev_adapter_width < 1:
                raise ValueError(f"ev_adapter_k/width must be >= 1, got "
                                 f"{self.ev_adapter_k}/{self.ev_adapter_width}")
        # Zero-reset inference diagnostic (scratch/evform/PREREG.md activity
        # check 3): set True to run the EXACT trained model with the adapter
        # branch removed. Not saved, not a training flag.
        self.ev_adapter_bypass = False
        self._ev_route_pi = None
        # --evmil-prior (scratch/fracprior/PREREG.md, scratch/priorpartner/
        # PREREG.md): which clinical reliability prior INITIALISES ev_R. ev_R
        # is an nn.Parameter saved in the state dict, so this is an init-only
        # flag: serving rebuilds load the trained ev_R and the kernel needs no
        # port. "clinical" is the shipped prior and its R is byte-identical to
        # every pre-existing arm's. Validated here (refuse-not-skip) even when
        # evidence_mil is off, so a typo can never sit silently in ck["args"].
        if evmil_prior not in self.EVMIL_PRIORS:
            raise ValueError(
                f"evmil_prior={evmil_prior!r} is not one of "
                f"{sorted(self.EVMIL_PRIORS)}")
        self.evmil_prior = str(evmil_prior)
        # --evmil-tau-init / --evmil-dif-init (scratch/evpool/PREREG.md): the
        # FIXED-AGGREGATOR flavour. ev_tau and ev_mix are saved nn.Parameters
        # that were MEASURED not to move off their init on the served seed-1
        # checkpoints (tau 0.92-1.01 vs init 1.0; softmax(ev_mix) 0.78/.11/.11
        # vs init [2,0,0]), so the init IS the aggregator. Init only: no effect
        # without --evidence-mil, and serve-inert (a rebuild loads the trained
        # ev_tau / ev_mix, and the kernel's forward reads them from the state
        # dict). Defaults 1.0 / "mean" reproduce the pre-flag init exactly.
        # Validated here refuse-not-skip even when evidence_mil is off, so a
        # typo can never sit silently in ck["args"].
        if evmil_dif_init not in self.EVMIL_DIF_INITS:
            raise ValueError(
                f"evmil_dif_init={evmil_dif_init!r} is not one of "
                f"{sorted(self.EVMIL_DIF_INITS)}")
        evmil_tau_init = float(evmil_tau_init)
        if not (evmil_tau_init > 0.0) or not math.isfinite(evmil_tau_init):
            raise ValueError(f"evmil_tau_init must be a finite positive float, "
                             f"got {evmil_tau_init!r}")
        self.evmil_tau_init = evmil_tau_init
        self.evmil_dif_init = str(evmil_dif_init)
        if self.evidence_mil:
            # The arm exists to change ONE thing (the aggregator). Anything
            # that attaches to the study transformer, its queries or the
            # pooled study vector has no meaning here; fail at build time
            # rather than training a silently different object.
            _bad = [n for n, v in (("study_pool", study_pool != "route"),
                                   ("flat_attn", bool(flat_attn)),
                                   ("cell_attn", bool(cell_attn)),
                                   ("band_pool", bool(band_pool)),
                                   ("routing_prior", bool(routing_prior)),
                                   ("query_family", query_family != "none"),
                                   ("qsa", qsa != "none"),
                                   ("text_dim", bool(text_dim)),
                                   ("n_report_aux", bool(n_report_aux)),
                                   ("moaks_blocks", bool(moaks_blocks)),
                                   ("aux_seg_classes", bool(aux_seg_classes)),
                                   ) if v]
            if _bad:
                raise ValueError(
                    "--evidence-mil replaces the study transformer and its "
                    "queries, so it is incompatible with: " + ", ".join(_bad))
        # --rubric-pool (scratch/rubpool/PREREG.md): a THIRD aggregation
        # inductive bias next to the study transformer (T1) and EvidenceMIL
        # (EV). Per-slice per-finding evidence is pooled by the operator the
        # published grading criterion NAMES, and every one of those operators
        # is FIXED: no learned attention, no learned temperature, no learned
        # reliability table. The only parameters the head adds are the 12-way
        # evidence projection (rub_head.*) and a per-finding bias (rub_b) - see
        # RUBPOOL_* below for the operator table. study_enc and its queries are
        # NOT constructed (this flag REPLACES them, gated residual branches are
        # a closed axis); the per-series aux head is KEPT UNCHANGED, which is
        # the deliberate difference from evidence_mil (which repurposed it).
        # Off by default and the off path is byte-identical: nothing below runs
        # and no module/buffer/parameter is added. Stored in ck["args"]; every
        # consumer that rebuilds from a checkpoint MUST read
        # args["rubric_pool"] (the weights also carry rub_head.* / rub_b, and
        # src/ckpt_build.py cross-checks both directions).
        self.rubric_pool = bool(rubric_pool)
        if self.rubric_pool:
            if self.evidence_mil:
                raise ValueError(
                    "--rubric-pool and --evidence-mil are two DIFFERENT "
                    "replacements for the same aggregator and cannot both be "
                    "on; the comparison between them is a package read, not a "
                    "model")
            if self.pfrel != "off" or self.bone != "off":
                raise ValueError("--rubric-pool replaces the study aggregator "
                                 "and cannot run with the --pfrel / "
                                 "--bonewsrpn specialist paths")
            _badr = [n for n, v in (("study_pool", study_pool != "route"),
                                    ("flat_attn", bool(flat_attn)),
                                    ("cell_attn", bool(cell_attn)),
                                    ("band_pool", bool(band_pool)),
                                    ("stack_depth", bool(stack_depth)),
                                    ("slice3d", slice3d not in (None, "none")),
                                    ("routing_prior", bool(routing_prior)),
                                    ("query_family", query_family != "none"),
                                    ("qsa", qsa != "none"),
                                    ("bypass_study_tr", bool(bypass_study_tr)),
                                    ("text_dim", bool(text_dim)),
                                    ("n_report_aux", bool(n_report_aux)),
                                    ("moaks_blocks", bool(moaks_blocks)),
                                    ("aux_seg_classes", bool(aux_seg_classes)),
                                    ) if v]
            if _badr:
                raise ValueError(
                    "--rubric-pool replaces the study transformer and its "
                    "queries, so it is incompatible with: " + ", ".join(_badr))
        # --evidence-mil-v2 (scratch/evmil_v2/PREREG.md): V2a adds a 32d
        # MORPHOLOGY RESIDUAL read at V1's own evidence locations. V1 compresses
        # each slice token to ONE scalar per finding before any aggregation;
        # the residual keeps 32 dimensions of the slices V1 already selected
        # (alpha is a DETACHED softmax of V1's smoothed evidence, so V1 stays
        # the localisation authority) and adds gamma_f * MLP(m_f) to the focal
        # findings' evidence. gamma is zero-init, so at step 0 the arm is
        # BIT-IDENTICAL to V1 (scratch/evmil_v2/equiv_check.py is the gate).
        # "mean" is the control arm: same parameters, uniform alpha, which tests
        # whether the gain (if any) needs V1's localisation or is just capacity.
        # evw and mean have IDENTICAL parameter sets, so the mode CANNOT be
        # sniffed from a state_dict - consumers must read
        # ck["args"]["evidence_mil_v2"] (src/ckpt_build.py refuses otherwise).
        self.ev2_mode = str(evidence_mil_v2 or "off")
        if self.ev2_mode not in ("off", "evw", "mean"):
            raise ValueError(f"--evidence-mil-v2 must be off|evw|mean, got "
                             f"{self.ev2_mode!r}")
        self.ev2_gamma_init = float(ev2_gamma_init)
        if self.ev2_mode != "off" and not self.evidence_mil:
            raise ValueError("--evidence-mil-v2 is a residual ON V1's evidence "
                             "path and requires --evidence-mil")
        # --syno-specialist (scratch/synov2/PREREG.md Amendment 1): a BUNDLED
        # Synovitis-only residual (burden B + morphology C + effusion context D)
        # on top of V1's study Synovitis logit. Refuse-not-skip: it is a residual
        # ON V1's evidence path, it re-uses the V2a residual machinery on one
        # DIFFUSE row (V2a covers the seven FOCAL rows and would fight it for the
        # same tokens), and --bypass-study-transformer is V1's attribution
        # control, which has no evidence path at all.
        self.syno_spec = bool(syno_specialist)
        self.syno_gamma_init = float(syno_gamma_init)
        # Evaluation-only decomposition switch; never read during training.
        self.syno_ablate = "none"
        if self.syno_spec:
            if not self.evidence_mil:
                raise ValueError("--syno-specialist is a residual ON V1's "
                                 "evidence path and requires --evidence-mil")
            if self.ev2_mode != "off":
                raise ValueError("--syno-specialist and --evidence-mil-v2 "
                                 "cannot be on together: V2a already adds a "
                                 "morphology residual read from the SAME slice "
                                 "tokens, so the two arms are not separable")
            if bypass_study_tr:
                raise ValueError("--syno-specialist requires --evidence-mil, "
                                 "which --bypass-study-transformer excludes")
        # --aux-seg-w (scratch/auxdice/PREREG.md): 0 (default) builds no
        # decoder at all, so state_dict and every forward path are byte-
        # identical to a checkpoint from before this flag existed. want_seg
        # is a separate, PARAMETER-FREE train-time switch (gramesc/
        # want_trunk precedent) so a checkpoint's contract still cannot be
        # sniffed from the state_dict alone - callers must read
        # ck["args"]["aux_seg_w"].
        self.aux_seg_decoder = None
        self.want_seg = False
        self._seg_logits = None
        # MOAKS subregional aux heads (--moaks-aux, scratch/latoarescue):
        # block sizes for the 62-column per-subregion target vector, e.g.
        # (14, 14, 12, 15, 4, 3). Parsed up front because the count feeds the
        # StudyEncoder's auxiliary-query slot below. Mutually exclusive with
        # report-aux: both attach at that slot and the trainer reads the
        # resulting logits from out[-1].
        self.moaks_blocks = tuple(int(b) for b in (moaks_blocks or ()))
        if self.moaks_blocks and n_report_aux:
            raise ValueError("--moaks-aux and --report-aux are mutually "
                             "exclusive: both use the auxiliary-query slot "
                             "and out[-1]")
        # Factorized trauma pair (see osseous_factorize). PARAMETER-FREE, so
        # state-dict sniffing cannot detect it - every consumer that rebuilds
        # a model from a checkpoint must read args["osseous_factor"] instead.
        self.osseous_factor = osseous_factor
        # Compartment-gated OA triad (see oa_gate_factorize) - same contract.
        self.oa_gate = oa_gate
        # --gram-escrow tap. When True, forward() stashes the per-study masked
        # mean of the BACKBONE slice features on self._trunk_rep. Default False
        # so no existing arm allocates it or changes by a single op; the
        # forward's return signature is deliberately NOT touched (scratch/
        # gramesc/PREREG.md 10). PARAMETER-FREE, so like osseous_factor it is
        # invisible to state-dict sniffing - but it is train-time only and
        # never affects inference, so it is not a checkpoint-contract key.
        self.want_trunk = False
        self._trunk_rep = None
        import timm
        self.slice_ctx = slice_ctx
        self.stack_depth = stack_depth
        # --backbone r3d_18 (scratch/basefam/PREREG.md arm V) only has a
        # forward that makes sense on the --stack-depth reshape (B*S, Z, H,
        # W); without stack_depth the backbone would receive the ordinary
        # (B*S*Z, k, H, W) slice-window tensor, which is not a volume. Fail
        # loudly at build time rather than silently training garbage.
        if backbone == "r3d_18" and not stack_depth:
            raise ValueError("--backbone r3d_18 requires --stack-depth "
                             "(and --slice-ctx == --n-slices, e.g. 24); its "
                             "forward consumes the stacked-depth (B*S, Z, H, "
                             "W) tensor as a 3D volume.")
        self.backbone = build_backbone(backbone, pretrained, slice_ctx,
                                       drop_path=drop_path)
        # --slice3d (scratch/slice3d/PREREG.md): through-plane mixing inside
        # the encoder, see Slice3DConvDW. "none" (default) installs NOTHING -
        # the module graph, state_dict keys and every forward op are the
        # pre-existing ones by construction. Installed AFTER build_backbone so
        # the pretrained weights land on the wrapped (centre) conv. Parameter-
        # bearing only for "inflate", but the "shift" variant is PARAMETER-
        # FREE and so invisible to state-dict sniffing: consumers must read
        # ck["args"]["slice3d"] (osseous_factor precedent).
        self.slice3d = str(slice3d or "none")
        if self.slice3d not in ("none", "shift", "inflate"):
            raise ValueError(f"slice3d={self.slice3d!r}")
        self._slice3d_mods = (install_slice3d(self.backbone, self.slice3d)
                              if self.slice3d != "none" else [])
        # --blurpool (scratch/regbundle/PREREG.md): anti-aliased downsampling,
        # see BlurConv2d. 0 (default) installs NOTHING - module graph,
        # state_dict and every forward op are the pre-existing ones by
        # construction. Parameter-free and state_dict-invisible: consumers
        # must read ck["args"]["blurpool"].
        self.blurpool = int(blurpool or 0)
        self._blurpool_mods = (install_blurpool(self.backbone)
                               if self.blurpool else [])
        # --mmix-p / --mmix-alpha (scratch/regbundle/PREREG.md): manifold
        # mixup at the last hidden layer (the 12 scored study-query vectors,
        # pre-head). TRAIN-ONLY, PARAMETER-FREE, set by the trainer after
        # construction (want_seg precedent); never touches inference and is
        # not a checkpoint-contract key. bs=1 has no in-batch partner, so
        # the partner is a DETACHED copy of the previous training study's
        # query tensor (cross-step bank); the trainer keeps the matching
        # target/weight bank and calls mmix_logits() once per step. Nothing
        # here is saved in the checkpoint.
        self.mmix_p = 0.0
        self.mmix_alpha = 0.4
        self._mmix_rng = random.Random(0)
        self._mmix_z = None
        self._mmix_bank = None
        # scratch/recipe2 --pcol-w: stash the 12 pre-head query vectors for the
        # trainer's auxiliary prototype loss. Off by default; not saved.
        self.pcol_on = False
        self._pcol_z = None
        # --dismix-p (scratch/dismix/PREREG.md, src/dismix.py): same train-only
        # parameter-free stash pattern as mmix; off (default) never stashes.
        self.dismix_on = False
        self._dismix_z = None
        dim = self.backbone.num_features
        if aux_seg_classes > 0:
            self.aux_seg_decoder = AuxSegHead(dim, aux_seg_classes)
        # The volumes arrive as uint8/255, i.e. mean~0.31 std~0.26, but the
        # pretrained backbone was trained on ImageNet-standardised input. Feeding
        # it [0,1] leaves first-layer activations ~4x too small and offset, which
        # wastes most of the transfer. Kept inside the model so training and the
        # Kaggle notebook cannot drift apart; non-persistent so the state_dict
        # (and the checkpoint) is unchanged.
        # `[0.485,0.456,0.406][:slice_ctx] or [0.449]*slice_ctx` looks like it
        # handles any width and does not: for slice_ctx=5 the slice returns all
        # THREE elements, which is truthy, so the `or` never fires and the
        # buffer is (1,3,1,1) against a 5-channel input. That is a shape error
        # on the first batch, i.e. after the job has queued and started.
        # Exactly 3 keeps the ImageNet per-channel stats so every existing
        # checkpoint is bit-identical; anything else uses the greyscale mean,
        # which is the honest choice anyway - these channels are adjacent
        # slices, not colour, and timm folds the pretrained RGB stem down to
        # in_chans itself.
        # --img-norm (scratch/win1_20261013/LAUNCH_20261013_late.md, ev2sfix): "half" = mean/std 0.5 per
        # channel, the tf_efficientnetv2_*/inception-style pretrained cfg (input in [-1, 1]). Non-persistent
        # buffer -> NO state-dict signature: ck["args"]["img_norm"] is the only contract (ckpt_build resolves
        # it; the Kaggle kernel must refuse it until ported). Default "imagenet" = byte-identical.
        if img_norm not in ("imagenet", "half"):
            raise ValueError(f"img_norm={img_norm!r} not in ('imagenet', 'half')")
        self.img_norm = img_norm
        if img_norm == "half":
            m = torch.full((slice_ctx,), 0.5)
            s = torch.full((slice_ctx,), 0.5)
        elif slice_ctx == 3:
            m = torch.tensor([0.485, 0.456, 0.406])
            s = torch.tensor([0.229, 0.224, 0.225])
        else:
            m = torch.full((slice_ctx,), 0.449)
            s = torch.full((slice_ctx,), 0.226)
        self.register_buffer("img_mean", m.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("img_std", s.view(1, -1, 1, 1), persistent=False)
        self.dim = dim
        # --xfuse (scratch/xfuse/PREREG.md): same-voxel FS/non-FS early fusion.
        # The loader emits (B, S, 2Z, H, W): rows [0, Z) the series, rows
        # [Z, 2Z) its geometry-matched partner on the same slice grid (exactly
        # zero when absent). The ConvNeXt patch-stem conv becomes 2*slice_ctx
        # input channels: [0.5*W_pre, 0.5*W_pre], bias unchanged, so the
        # initial stem output is IDENTICAL to the pretrained one when the
        # partner duplicates the series. A zero-init nn.Embedding(2, dim) on
        # the per-slice feature carries the pair-present flag (centre-channel
        # partner present), the learned "unpaired" embedding. False (default)
        # constructs nothing: state_dict keys and forward are unchanged.
        self.xfuse = bool(xfuse)
        if self.xfuse and (stack_depth or blurpool or slice3d != "none"
                           or pfrel != "off" or bone != "off"):
            raise ValueError("--xfuse is not combined with stack_depth/blurpool/"
                             "slice3d/pfrel/bonewsrpn (not implemented)")
        if self.xfuse:
            _stem = getattr(self.backbone, "stem", None)
            if _stem is None or not isinstance(_stem[0], nn.Conv2d):
                raise ValueError("--xfuse needs a ConvNeXt-style backbone.stem[0] conv")
            _c = _stem[0]
            _n = nn.Conv2d(2 * _c.in_channels, _c.out_channels, _c.kernel_size,
                           stride=_c.stride, padding=_c.padding,
                           bias=_c.bias is not None)
            with torch.no_grad():
                _n.weight.copy_(torch.cat([0.5 * _c.weight, 0.5 * _c.weight], 1))
                if _c.bias is not None:
                    _n.bias.copy_(_c.bias)
            _stem[0] = _n
            self.xfuse_emb = nn.Embedding(2, dim)
            nn.init.zeros_(self.xfuse_emb.weight)
        # --pfrel: the relational PF-OA specialist reads the backbone's
        # RETAINED (N, dim, 12, 12) feature map (no global pool, spec s2).
        self.pfrel_net = (None if self.pfrel == "off" else PFRelNet(
            dim, mode=self.pfrel, lambda_pos=pfrel_lambda_pos,
            beta=pfrel_beta, drop=drop))
        # --bonewsrpn: the osseous specialist needs BOTH the stage-3 (24x24)
        # and the stage-4 (12x12) maps. stage 4 is what forward_features
        # returns; stage 3 is captured by a forward hook that keeps the graph
        # (NOT detached - gradient must reach stages.2 once it unfreezes).
        self._bone_c3 = None
        self.bone_net = None
        if self.bone != "off":
            _st = getattr(self.backbone, "stages", None)
            if _st is None or len(_st) < 4:
                raise ValueError("--bonewsrpn needs a 4-stage backbone with "
                                 ".stages (convnext_tiny)")
            _ch = None
            try:
                _ch = int(self.backbone.feature_info.channels()[2])
            except Exception:
                _ch = None
            if _ch is None:
                # timm's ConvNeXt built with num_classes=0 may carry no
                # feature_info; take the channel count from a 1-sample dry run
                # of stem + stages 0-2 instead of hardcoding 384.
                try:
                    with torch.no_grad():
                        _t = torch.zeros(1, slice_ctx, 64, 64)
                        _t = self.backbone.stem(_t)
                        for _i in range(3):
                            _t = _st[_i](_t)
                        _ch = int(_t.shape[1])
                except Exception:
                    _ch = None
            if _ch is None:
                raise ValueError("--bonewsrpn cannot resolve the stage-3 "
                                 "channel count from feature_info")
            _st[2].register_forward_hook(
                lambda m, i, o: setattr(self, "_bone_c3", o))
            self.bone_net = BoneWSRPN(
                _ch, dim, dim=128, mode=self.bone, k=self.bone_k,
                tau=self.bone_tau, shell=self.bone_shell, drop=drop)
        # ---- banded spatial pooling -------------------------------------
        # The backbone returns a GLOBALLY pooled vector per slice, so the
        # within-slice left-right axis is averaged away - and on coronal and
        # axial that axis IS medial vs lateral, which is our largest measured
        # gap (medial findings beat their lateral twins by 0.12-0.13 in 4/4
        # folds). RSNA 2025's 6th place hit the identical problem across 13
        # anatomical locations and solved it with no localisation labels at
        # all: collapse the axis that is not left-right, split the remaining
        # one into overlapping left / centre / right bands, concatenate.
        #
        # The global vector is KEPT as a fourth input, and band_proj is
        # initialised as identity on it and ZERO on the three bands. So at step
        # 0 this module is bit-identical to the model without it, and the bands
        # can only be used if they earn it - the zero-init residual trick. That
        # matters because a from-scratch projection in front of a pretrained
        # trunk otherwise destroys the transfer on the first batch.
        #
        # Note the band semantics differ by plane: on coronal and axial the
        # column axis is medial-lateral, on sagittal it is anterior-posterior.
        # Both are informative and the series-type embedding downstream can
        # tell them apart, so this is not a defect - but it is why the flag is
        # worth measuring rather than assuming.
        self.band_delta = band_delta
        self.band_norm = nn.LayerNorm(dim) if band_pool else None
        self.band_proj = nn.Linear(4 * dim, dim) if band_pool else None
        if self.band_proj is not None:
            with torch.no_grad():
                self.band_proj.weight.zero_()
                self.band_proj.bias.zero_()
                self.band_proj.weight[:, 3 * dim:].copy_(torch.eye(dim))
        self.series_enc = SeriesEncoder(dim, drop=drop, n_pos=n_slice_pos,
                                        n_pos_types=slice_pos_types,
                                        mode=slice_enc)
        self.study_enc = None if (self.evidence_mil
                                  or self.rubric_pool) else StudyEncoder(
                                      dim, n_out=n_out, drop=drop, n_seq=n_seq,
                                      study_pool=study_pool,
                                      routing_prior=routing_prior,
                                      flat_attn=flat_attn, cell_attn=cell_attn,
                                      gate_init=gate_init,
                                      cell_slice_bands=cell_slice_bands,
                                      n_aux=n_report_aux + len(self.moaks_blocks),
                                      query_family=query_family,
                                      qsa=qsa, qsa_gate_init=qsa_gate_init,
                                      flat_queries=flat_queries,
                                      bypass_study_tr=bypass_study_tr)
        # --qsa (scratch/qsa/PREREG.md): recorded here for the trainer's
        # banner; the module itself lives in StudyEncoder. Saved in
        # ck["args"]; consumers must rebuild with it.
        self.qsa = qsa if (self.evidence_mil
                           or self.rubric_pool) else self.study_enc.qsa
        self.flat_attn = bool(flat_attn)
        # ---- how the HxW feature map is reduced to one vector per slice ------
        # The backbone GLOBALLY AVERAGES a 12x12 map. The published OA criterion
        # is a ">=1 cm" lesion, which at 0.42 mm/px covers roughly one native
        # cell - so a positive study's evidence is averaged with 143 cells of
        # normal tissue before any finding query exists. That is a DILUTION
        # problem, distinct from the ADDRESSING problem --cell-attn attacks, and
        # it has never been tested. (The AttnPool dilution result was about the
        # SLICE axis - effective 3.73 of 24 - and says nothing about within-slice
        # dilution at 1/144.)
        #
        # Generalised mean: f = (mean |x|^p * sign)^(1/p). p=1 is the average,
        # p->inf is the max. Standard GeM clamps to a positive epsilon before
        # pow(), which is wrong here: ConvNeXt features after LayerNorm are
        # signed and roughly half of them are negative, so clamping would
        # discard that half AND destroy the p=1 control. The SIGNED form below
        # is exactly the identity at p=1, which is what makes a clean control
        # possible - and the p=1 path skips the arithmetic entirely so the
        # control is bit-identical rather than merely equal in exact maths.
        self.gem_p = float(gem_p)
        self.cell_k = int(cell_attn)
        # ---- which stage the cell tokens are tapped from --------------------
        # At 384px with ~0.39 mm/px, the FINAL ConvNeXt stage is 12x12, so one
        # native cell is 32px ~ 12.5 mm - and the published OA criterion is a
        # ">=1 cm" lesion, i.e. smaller than one cell. Stage 2 is 24x24, so its
        # native cell is 6.25 mm. Tapping it is strictly cheaper than a crop
        # stream, which buys the same cells-per-lesion by UPSAMPLING (pixel
        # spacing is already near native, so a crop adds no information).
        #
        # Grabbed with a forward hook rather than by reimplementing timm's
        # forward, which would be brittle across backbones and versions. The
        # hook's output must stay attached to the autograd graph under gradient
        # checkpointing - that is verified by execution, not assumed.
        self.cell_stage = int(cell_stage)
        self._cell_feat = None
        self.cell_proj = nn.Identity()
        if self.cell_k and self.cell_stage >= 0:
            stages = getattr(self.backbone, "stages", None)
            if stages is None:
                raise ValueError(
                    f"--cell-stage needs a backbone with .stages; "
                    f"{type(self.backbone).__name__} has none.")
            if self.cell_stage >= len(stages):
                raise ValueError(f"--cell-stage {self.cell_stage} but the "
                                 f"backbone has {len(stages)} stages")
            stages[self.cell_stage].register_forward_hook(
                lambda m, i, o: setattr(self, "_cell_feat", o))
            # Probe the channel count rather than hardcoding it: stage dims
            # differ per backbone and per stage (ConvNeXt stage 2 is 384, the
            # final stage 768), and a wrong guess is a shape error after the
            # job has queued.
            with torch.no_grad():
                self.backbone(torch.zeros(1, slice_ctx, 64, 64))
            c = self._cell_feat.shape[1]
            self._cell_feat = None
            if c != dim:
                self.cell_proj = nn.Linear(c, dim)
        # One scalar head per finding, reading only that finding's routed vector.
        if self.ps25:
            if evidence_mil or stack_depth or band_pool or cell_attn or osseous_factor or oa_gate \
                    or flat_attn or rubric_pool or xfuse or str(bone) != "off" or str(pfrel) != "off":
                raise ValueError("--ps25 is only implemented on the plain path "
                                 "(no evmil/stack-depth/band/cell/flat/osseous/oa-gate/rubric/xfuse/bone/pfrel)")
            self.ps25_type = nn.Embedding(16, 2 * dim)
            nn.init.zeros_(self.ps25_type.weight)
            self.ps25_head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Dropout(drop),
                                           nn.Linear(2 * dim, n_out))
        # scratch/seqadapt: --seq-adapt R (rank; 0 = off = nothing constructed,
        # no hook, forward byte-identical). Per-series-type FiLM + rank-R
        # residual adapter on every ConvNeXt stage output (_SeqAdaptStage).
        self.seq_adapt = int(seq_adapt)
        if self.seq_adapt:
            if stack_depth or xfuse or cell_attn or band_pool or bn_valid_only \
                    or str(bone) != "off" or str(pfrel) != "off" \
                    or not hasattr(self.backbone, "stages") or "convnext" not in str(backbone):
                raise ValueError("--seq-adapt is only implemented on the ConvNeXt "
                                 "plain/osseous path (no stack-depth/xfuse/cell/band/"
                                 "bn-valid-only/bone/pfrel)")
            self._sa_ctx = {"rows": None}
            _chs = [int(fi["num_chs"]) for fi in self.backbone.feature_info]
            if len(_chs) != len(self.backbone.stages):
                raise ValueError(f"--seq-adapt: feature_info {_chs} vs "
                                 f"{len(self.backbone.stages)} stages")
            self.seq_adapt_mods = nn.ModuleList(
                _SeqAdaptStage(c, 6, self.seq_adapt, self._sa_ctx, seed=1009 + i)
                for i, c in enumerate(_chs))
            for st, m in zip(self.backbone.stages, self.seq_adapt_mods):
                st.register_forward_hook(m.hook)
        self.drop = nn.Dropout(drop)
        self.head = nn.Parameter(torch.zeros(n_out, dim))
        self.head_b = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.head, std=0.02)
        # Auxiliary per-series head: supervising each series with the study
        # label is a multiple-instance signal that stabilises early training,
        # when the study transformer has nothing useful to attend over yet.
        self.aux_head = None if self.evidence_mil else nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(drop), nn.Linear(dim, n_out))
        if self.evidence_mil:
            self._build_evidence_mil(dim, n_out, drop)
        if self.rubric_pool:
            self._build_rubric_pool(dim, n_out, drop)
        # Optional report-text head: regress the study representation onto a
        # sentence embedding of the whole report. Twelve ordinal grades throw
        # away laterality, compartment, severity wording and every unscored
        # finding; this puts that back as a training signal. Never used at
        # inference, so it costs nothing at test time.
        self.text_head = (nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, text_dim))
                          if text_dim else None)
        # Auxiliary REPORT targets (src/extract_aux.py): compartment-localised
        # cartilage, marrow oedema, osteophytes, LCL/PLC, extensor mechanism,
        # loose bodies. Eight findings the competition does not score, so this
        # is pure extra supervision for the shared trunk - and three of them
        # carry the compartment signal the weak findings need. Training-only:
        # test.csv has no Report column, so inference never calls this head.
        # One weight vector per auxiliary target, read off ITS OWN query - the
        # same form as `head` for the scored findings. Replaces the old
        # Linear(dim, n_report_aux) over the mean of the twelve scored queries,
        # whose gradient was identical for every finding by construction.
        self.n_report_aux = int(n_report_aux)
        if n_report_aux:
            self.aux_w = nn.Parameter(torch.zeros(n_report_aux, dim))
            self.aux_b = nn.Parameter(torch.zeros(n_report_aux))
            nn.init.normal_(self.aux_w, std=0.02)
        else:
            self.aux_w = self.aux_b = None
        # MOAKS subregional aux heads: ONE LINEAR HEAD PER AUX BLOCK, each
        # reading its OWN auxiliary query out of the study transformer - the
        # same attachment point as the report-aux head above, and the same
        # cure for the mean-pool gradient degeneracy its comment documents
        # (per-block queries give each family its own routed vector; within a
        # block the columns are separate rows of that block's Linear).
        # Training-only, like report-aux: dropped at serve time, so the scored
        # path and runtime are untouched (scratch/latoarescue/PREREG.md).
        if self.moaks_blocks:
            self.moaks_heads = nn.ModuleList(
                [nn.Linear(dim, b) for b in self.moaks_blocks])
        else:
            self.moaks_heads = None
        self.report_aux = None   # legacy attribute; kept so old code paths
                                 # that test `is not None` stay correct
        # --ev-adapter is constructed LAST, after every other module, so the
        # global RNG stream consumed by the pre-existing modules is unchanged
        # and an ev_adapter run shares its init with the control seed-for-seed.
        self.ev_adapter_mod = None
        if self.ev_adapter != "off":
            self.ev_adapter_mod = EvAdapter(dim, self.ev_adapter,
                                            self.ev_adapter_k,
                                            self.ev_adapter_width)

    # ---- evidence-MIL head (scratch/evmil/PREREG.md + Amendment 2) --------
    # FOCAL findings are a handful of slices in one or two series; DIFFUSE
    # findings are a property of the whole joint. The shipped aggregator gives
    # both the same unconstrained machinery (attention pool over slices, then a
    # transformer plus a free cross-attention query over series). Here the
    # aggregation is CONSTRAINED and named: log-sum-exp over slices for focal,
    # mean over slices for diffuse, then a per-finding reliability weighting
    # over the present series that is initialised from the clinical prior of
    # which plane/sequence actually shows the finding.
    EVMIL_FOCAL = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                   "Fracture", "Contusion", "Baker\'s")
    EVMIL_DIFFUSE = ("Medial OA", "Lateral OA", "PF OA", "Effusion",
                     "Synovitis")
    # Reliability prior, as LOGITS over the six (plane, fluid) series types,
    # softmax-normalised over the series a study actually has. Learnable.
    # Keys are (plane_prefix, fluid) with fluid None = both.
    EVMIL_PRIOR = {
        "ACL": {("Sag", 1): 2.0, ("Sag", 0): 1.5, ("Cor", None): 1.0,
                ("Ax", None): 0.0},
        "MCL": {("Cor", None): 2.0, ("Ax", None): 1.0, ("Sag", None): 0.5},
        "Medial Meniscus": {("Sag", None): 2.0, ("Cor", None): 1.5,
                            ("Ax", None): 0.0},
        "Lateral Meniscus": {("Sag", None): 2.0, ("Cor", None): 1.5,
                             ("Ax", None): 0.0},
        "Fracture": {(None, 1): 2.0, (None, 0): 0.5},
        "Contusion": {(None, 1): 2.0, (None, 0): 0.5},
        "Baker\'s": {("Sag", None): 1.5, ("Ax", None): 1.5,
                     ("Cor", None): 0.5},
    }
    # Alternative priors, selected by --evmil-prior. ev_R is a learnable
    # nn.Parameter that a 2026-09-20 read (scratch/evwave/evR_read/RESULTS.md)
    # showed never moves off its init, so the prior IS the model's series
    # routing and swapping it is a real mechanism change, not a nudge.
    #   fracbone (scratch/fracprior/PREREG.md): a cortical break is a
    #     NON-fat-sat T1/PD sign; the shipped clinical prior points Fracture at
    #     fluid-sensitive series (4:1), which is the marrow-oedema view. Flip
    #     it, keeping some fluid weight for the oedema. Contusion (oedema
    #     WITHOUT a discrete line, per the host rubric) is deliberately
    #     UNCHANGED, so the two osseous columns become attributable.
    #   confirm (scratch/priorpartner/PREREG.md): a confirmatory-plane prior -
    #     weight the plane a radiologist uses to CONFIRM the finding over the
    #     plane it is first seen on. Fracture and Contusion are UNCHANGED from
    #     clinical here so the fracbone arm and this arm are attributable.
    EVMIL_PRIORS = {
        "clinical": EVMIL_PRIOR,
        "fracbone": dict(EVMIL_PRIOR,
                         **{"Fracture": {(None, 0): 2.0, (None, 1): 1.0}}),
        "confirm": dict(EVMIL_PRIOR, **{
            "ACL": {("Cor", None): 2.0, ("Sag", 1): 1.5, ("Sag", 0): 1.0,
                    ("Ax", None): 1.0},
            "MCL": {("Ax", None): 2.0, ("Cor", None): 1.5,
                    ("Sag", None): 0.5},
            "Medial Meniscus": {("Cor", None): 2.0, ("Sag", None): 1.5,
                                ("Ax", None): 0.5},
            "Lateral Meniscus": {("Cor", None): 2.0, ("Sag", None): 1.5,
                                 ("Ax", None): 0.5},
            "Baker\'s": {("Ax", None): 2.0, ("Sag", None): 1.0,
                         ("Cor", None): 0.5},
        }),
    }

    # --evmil-dif-init: which of (mean, quantile-0.8, max) the diffuse slice
    # mixture STARTS dominated by. Logits, not weights; softmax([2,0,0]) is
    # 0.78/0.11/0.11, the shipped init.
    EVMIL_DIF_INITS = {"mean": (2.0, 0.0, 0.0),
                       "quantile": (0.0, 2.0, 0.0),
                       "max": (0.0, 0.0, 2.0)}

    def _build_evidence_mil(self, dim, n_out, drop):
        # Imported lazily: model.py has never depended on data.py and the
        # inference notebook inlines these modules separately. Only the arm
        # pays the import.
        from data import FINDINGS, SERIES_TYPES
        if n_out != len(FINDINGS):
            raise ValueError(f"--evidence-mil needs the 12 scored findings, "
                             f"got n_out={n_out}")
        foc, dif = set(self.EVMIL_FOCAL), set(self.EVMIL_DIFFUSE)
        if foc | dif != set(FINDINGS) or foc & dif:
            raise ValueError("focal/diffuse sets must PARTITION data.FINDINGS")
        _types = list(SERIES_TYPES)
        if getattr(self, "tfcrop_types", 0):
            from data import TFCROP_PRIOR_AS
            if self.tfcrop_types != len(TFCROP_PRIOR_AS):
                raise ValueError(f"tfcrop_types={self.tfcrop_types} != {len(TFCROP_PRIOR_AS)}")
            _types += list(TFCROP_PRIOR_AS)
        n_types = len(_types)
        # Per-slice, per-finding evidence. LayerNorm first because the slice
        # transformer is norm_first and so returns UNNORMALISED tokens (the
        # aux head this replaces normalises for the same reason).
        self.ev_head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(drop),
                                     nn.Linear(dim, n_out))
        # Per-finding LSE temperature: tau -> 0 is max (one slice decides),
        # tau -> inf is the mean. Init --evmil-tau-init (default 1.0, which is
        # torch.ones and so byte-identical to every pre-flag arm), clamped at
        # 0.2 in the forward so a collapsing temperature cannot make the pool a
        # hard max on one slice.
        # Refuse-not-skip, mirroring the evmil_prior guard below: the ctor
        # validates too, but a direct _build_evidence_mil call (the smoke
        # harness, a future rebuild path) must not silently accept a bad init.
        _tau = float(self.evmil_tau_init)
        if not (_tau > 0.0) or not math.isfinite(_tau):
            raise ValueError(f"evmil_tau_init must be a finite positive float, "
                             f"got {self.evmil_tau_init!r}")
        self.ev_tau = nn.Parameter(torch.full((n_out,), _tau))
        self.register_buffer(
            "ev_focal", torch.tensor([f in foc for f in FINDINGS]),
            persistent=False)
        # Clinical reliability prior -> (n_out, n_types) logits.
        R = torch.ones(n_out, n_types)
        try:
            prior = self.EVMIL_PRIORS[self.evmil_prior]
        except KeyError:
            raise ValueError(
                f"evmil_prior={self.evmil_prior!r} is not one of "
                f"{sorted(self.EVMIL_PRIORS)}") from None
        # Refuse-not-skip: a prior keyed on a name that is not a scored finding
        # would silently contribute NOTHING (the loop below is over FINDINGS),
        # so a typo like "Bakers" must fail at build time, not at read time.
        _bad = sorted(set(prior) - set(FINDINGS))
        if _bad:
            raise ValueError(f"EVMIL_PRIORS[{self.evmil_prior!r}] has keys "
                             f"that are not data.FINDINGS: {_bad}")
        for i, f in enumerate(FINDINGS):
            rules = prior.get(f)
            if rules is None:
                continue                      # diffuse: all-ones, uniform
            for j, (plane, fluid) in enumerate(_types):
                for (kp, kf), v in rules.items():
                    if (kp is None or plane.startswith(kp)) and \
                            (kf is None or kf == fluid):
                        R[i, j] = v
        self.ev_R = nn.Parameter(R)
        self.ev_b = nn.Parameter(torch.zeros(n_out))
        # Amendment 3(2): NOTHING sits on top of the accumulation. The study
        # logit is b_f + sum_s r(f,s) e_pooled(f,s) and nothing else - no
        # per-finding MLP, no family adapters (Amendment 3(1): task-group
        # adapters are a measured kill, qsa; including one would confound the
        # aggregation assay with a dead hypothesis).
        # Amendment 3(3): DIFFUSE slice pooling is a learned mixture of
        # mean / quantile-0.8 / max, one softmax over three logits per diffuse
        # finding, initialised [2, 0, 0] so step 0 is the plain mean. Only the
        # diffuse rows exist, so the arm carries no dead parameters.
        self.register_buffer(
            "ev_dif_idx",
            torch.tensor([i for i, f in enumerate(FINDINGS) if f in dif]),
            persistent=False)
        try:
            _dif0 = self.EVMIL_DIF_INITS[self.evmil_dif_init]
        except KeyError:
            raise ValueError(
                f"evmil_dif_init={self.evmil_dif_init!r} is not one of "
                f"{sorted(self.EVMIL_DIF_INITS)}") from None
        self.ev_mix = nn.Parameter(
            torch.tensor(list(_dif0)).expand(len(dif), 3).clone())
        if self.ev2_mode != "off":
            self._build_evidence_mil_v2(dim)
        if self.syno_spec:
            self._build_syno_specialist(dim)

    def _build_evidence_mil_v2(self, dim):
        """V2a morphology residual (scratch/evmil_v2/PREREG.md).

        One projection + one 2-layer read-out + one gamma PER FOCAL FINDING, in
        data.FINDINGS order (ev2_foc_idx). Under --osseous-factor rows 10/11 are
        reinterpreted as the oedema (E) and fracture-line (L) evidence rows, and
        because Contusion (10) and Fracture (11) are BOTH focal they already own
        one head and one gamma each: the residual therefore enters E and L
        separately and BEFORE osseous_factorize, never the final Cont/Frac
        logits. The diffuse five get no residual at all (placebo columns).
        Parameter names start with ev2_ and do NOT start with "backbone.", so
        src/train.py's optimiser split puts them in the HEAD LR group, not BBLR.
        """
        self.register_buffer("ev2_foc_idx",
                             self.ev_focal.nonzero().flatten().clone(),
                             persistent=False)
        nf = int(self.ev2_foc_idx.numel())
        # z_{f,t} = GELU(LayerNorm(W_f h_t)), 768 -> 32.
        self.ev2_proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, 32), nn.LayerNorm(32), nn.GELU())
             for _ in range(nf)])
        # r_f = Linear(32->16) -> GELU -> Linear(16->1).
        self.ev2_head = nn.ModuleList(
            [nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
             for _ in range(nf)])
        # gamma_f init. 0 makes V2 == V1 bit-exactly, but the T4 dynamics probe
        # (scratch/evmil_v2/t4probe/RESULTS.md, 2026-09-15) measured exactly the
        # CLAUDE.md "zero-init gates collapse" failure: over ~183 optimiser
        # steps |gamma| random-walked between 8e-6 and 1.0e-3 and came BACK to
        # 4e-4 (0/7 above 1e-3), because the gradient sign flips and AdamW's
        # decoupled weight decay pulls a 1-D head parameter toward 0. PREREG
        # gate 3's named fallback is therefore in force: --ev2-gamma-init, default
        # 0.01 (in ck["args"] like gate_init/qsa_gate_init, so every run records
        # the init it used). V2a is then ~V1 at init, no longer exactly V1; the equivalence
        # gate is still measured, at gamma forced to 0
        # (scratch/evmil_v2/equiv_check.py), which is what proves the residual
        # enters the logit ONLY through gamma.
        self.ev2_gamma = nn.Parameter(
            torch.full((nf,), float(self.ev2_gamma_init)))

    # ---- rubpool (scratch/rubpool/PREREG.md) -------------------------------
    # The operator table. Each finding is pooled over SLICES within a series
    # and then over SERIES by the shape of its published criterion, and every
    # operator is a fixed function of the evidence - nothing here is learned.
    #   TOP2    "reaches the articular surface on >=2 images" -> mean of the
    #           two strongest slices in a series, then max over series.
    #   EXTENT  extent/volume criteria (>=1 cm of >50%-thickness cartilage
    #           loss, moderate+ effusion, moderate+ Baker's) -> a soft extent
    #           (log-sum-exp at a FIXED temperature, mean-normalised so it
    #           interpolates mean<-->max), then max over series.
    #   FOCAL   one-lesion criteria (>50% of ACL fibres, a high-grade acute
    #           MCL, an acute cortical break, oedema without a line) -> max
    #           over slices, then max over series.
    #   DIFFUSE Synovitis has NO severity threshold and is a whole-joint
    #           property -> mean over slices, then mean over the present
    #           series.
    RUBPOOL_TOP2 = ("Medial Meniscus", "Lateral Meniscus")
    RUBPOOL_EXTENT = ("Medial OA", "Lateral OA", "PF OA", "Effusion", "Baker's")
    RUBPOOL_FOCAL = ("ACL", "MCL", "Contusion", "Fracture")
    RUBPOOL_DIFFUSE = ("Synovitis",)
    # Fixed LSE temperature. A module CONSTANT, not an nn.Parameter and not a
    # buffer: the whole point of the arm is that the operator cannot be
    # learned away (evmil's ev_tau is the learnable version of exactly this
    # knob, and this arm exists to price the fixed one).
    RUBPOOL_TAU = 1.0

    def _build_rubric_pool(self, dim, n_out, drop):
        # Lazy import for the same reason _build_evidence_mil does it: model.py
        # has never depended on data.py and the inference notebook inlines the
        # modules separately.
        from data import FINDINGS
        if n_out != len(FINDINGS):
            raise ValueError(f"--rubric-pool needs the 12 scored findings, "
                             f"got n_out={n_out}")
        groups = (("top2", self.RUBPOOL_TOP2), ("ext", self.RUBPOOL_EXTENT),
                  ("foc", self.RUBPOOL_FOCAL), ("dif", self.RUBPOOL_DIFFUSE))
        named = [f for _, g in groups for f in g]
        # Refuse-not-skip: a typo in the table would silently drop a finding
        # into no operator at all (the buffers below are built by lookup), so
        # the partition is checked at build time.
        if sorted(named) != sorted(FINDINGS) or len(named) != len(set(named)):
            raise ValueError("RUBPOOL_* operator groups must PARTITION "
                             "data.FINDINGS exactly once each")
        idx = {f: i for i, f in enumerate(FINDINGS)}
        for key, g in groups:
            # persistent=False: these are a constant property of the operator
            # table, not trained state, so the state dict stays exactly
            # {rub_head.*, rub_b} and a rebuild cannot inherit a stale table.
            self.register_buffer(f"rub_{key}_idx",
                                 torch.tensor([idx[f] for f in g],
                                              dtype=torch.long),
                                 persistent=False)
        _tau = float(self.RUBPOOL_TAU)
        if not (_tau > 0.0) or not math.isfinite(_tau):
            raise ValueError(f"RUBPOOL_TAU must be finite positive, got {_tau}")
        # LayerNorm first: the slice transformer is norm_first and returns
        # UNNORMALISED tokens (the aux head this sits beside normalises for the
        # same reason).
        self.rub_head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(drop),
                                      nn.Linear(dim, n_out))
        self.rub_b = nn.Parameter(torch.zeros(n_out))

    def _rubric_forward(self, tokens, mask, B, S, Z):
        """tokens (B*S, Z, D) -> (logits (B,12), contribution weights (B,12,S)).

        No learned pooling: every reduction below is the fixed operator named
        in the table above.
        """
        n_out = self.rub_b.shape[0]
        e = self.rub_head(tokens).reshape(B, S, Z, n_out).float()   # (B,S,Z,F)
        # ---- over SLICES, within a series -> (B,S,F) ----
        ser = e.new_zeros(B, S, n_out)
        fi, ti = self.rub_foc_idx, self.rub_top2_idx
        xi, di = self.rub_ext_idx, self.rub_dif_idx
        ser[..., fi] = e[..., fi].max(2).values
        # "on >=2 images": the mean of the two strongest slices. A single hot
        # slice therefore cannot carry a meniscus tear on its own, which is
        # exactly what the criterion says. k is clamped for the Z==1 case.
        k = min(2, int(Z))
        ser[..., ti] = e[..., ti].topk(k, dim=2).values.mean(2)
        tau = float(self.RUBPOOL_TAU)
        ser[..., xi] = tau * (torch.logsumexp(e[..., xi] / tau, dim=2)
                              - math.log(float(Z)))
        ser[..., di] = e[..., di].mean(2)
        # ---- over SERIES -> (B,F) ----
        # No reliability weighting: this arm has no ev_R, so a finding is read
        # off whichever present series shows it best (max), except Synovitis,
        # which is averaged over the whole examination.
        m = mask.to(ser.dtype)                                      # (B,S)
        cnt = m.sum(1).clamp(min=1.0)                               # (B,)
        smax, amax = ser.masked_fill(m.unsqueeze(-1) == 0, -1e4).max(1)
        smean = (ser * m.unsqueeze(-1)).sum(1) / cnt.unsqueeze(-1)
        pooled = smax.clone()
        pooled[..., di] = smean[..., di]
        logits = self.rub_b + pooled                                # (B,F)
        # Contribution record in the (B, n_out, S) shape the trainer's routing
        # consumers expect: one-hot at the series the max came from, uniform
        # over the present series for the mean-pooled findings.
        w = ser.new_zeros(B, n_out, S)
        w.scatter_(2, amax.detach().unsqueeze(-1), 1.0)
        w[:, di] = (m / cnt.unsqueeze(-1)).unsqueeze(1).expand(
            B, int(di.numel()), S)
        return logits, w

    trauma_head = False

    def _enable_trauma_head(self):
        """--trauma-head: add the independent contusion factor K (scratch/traumahead/PREREG.md). Called AFTER construction
        (train.py after KneeNet(...).to(device); ckpt_build before load_state_dict). Refuse-not-skip on unsupported paths."""
        bad = []
        if not self.evidence_mil: bad.append("needs --evidence-mil")
        if not self.osseous_factor: bad.append("needs --osseous-factor (rows 10/11 = z_E/z_L)")
        if getattr(self, "evdrop", False): bad.append("evdrop")
        if getattr(self, "pat", False): bad.append("pat")
        if getattr(self, "syno_spec", False): bad.append("syno_specialist")
        if str(getattr(self, "ev2_mode", "off")) != "off": bad.append("evidence_mil_v2")
        if getattr(self, "darc_local", False): bad.append("darc_local")
        if getattr(self, "ev_adapter_mod", None) is not None: bad.append("ev_adapter")
        if bad:
            raise ValueError(f"--trauma-head unsupported with: {bad}")
        dev = self.ev_b.device
        dim = self.ev_head[2].in_features
        self.trauma_k_lin = nn.Linear(dim, 1).to(dev)
        self.trauma_k_tau = nn.Parameter(torch.full((1,), float(self.evmil_tau_init), device=dev))
        self.trauma_k_b = nn.Parameter(torch.zeros(1, device=dev))
        self.trauma_head = True

    def _trauma_k(self, tokens, attn, B, S, Z):
        """Independent contusion factor: (study logit (B,), per-series logit (B,S)). Same LSE pooling as a focal EV column;
        series weights = the Contusion column's reliability softmax (attn = r.permute(0,2,1), (B,F,S))."""
        x = self.ev_head[1](self.ev_head[0](tokens))
        ek = self.trauma_k_lin(x).reshape(B, S, Z, 1).float()
        es = ek.permute(0, 1, 3, 2).reshape(B * S, 1, Z)
        es = F.avg_pool1d(F.pad(es, (1, 1), mode="replicate"), 3, stride=1)
        es = es.reshape(B, S, 1, Z).permute(0, 1, 3, 2)
        tau = self.trauma_k_tau.clamp(min=0.2)
        lse = tau * (torch.logsumexp(es / tau, dim=2) - math.log(float(Z)))   # (B,S,1)
        r = attn[:, 10, :].unsqueeze(-1).to(lse.dtype)                     # (B,S,1)
        zk = self.trauma_k_b + (r * lse).sum(1)                             # (B,1)
        return zk[:, 0], lse[..., 0]

    def _evidence_forward(self, tokens, stype, mask, B, S, Z):
        """tokens (B*S, Z, D) -> (logits (B,12), per-series evidence (B,S,12),
        reliability weights (B,12,S))."""
        n_out = self.ev_b.shape[0]
        # --ev-adapter (scratch/evform): conditional evidence FORMATION, the
        # ONLY line this flag adds to the serve path. Off -> nothing runs.
        if self.ev_adapter_mod is not None and not self.ev_adapter_bypass:
            tokens = self.ev_adapter_mod(tokens)
            self._ev_route_pi = self.ev_adapter_mod.last_pi
        e = self.ev_head(tokens).reshape(B, S, Z, n_out).float()  # (B,S,Z,F)
        if self.darc_local:
            # scratch/shortcutarms §3: the TDE evidence REPLACES e.
            from darc import darc_rows
            _lin = self.ev_head[2]
            e = F.linear(self.ev_head[1](self.ev_head[0](tokens)),
                         darc_rows(_lin.weight, self.darc_keep),
                         _lin.bias).reshape(B, S, Z, n_out).float()
        # Amendment 2(1): 3-tap mean over neighbouring slices before the LSE,
        # edge-replicated. A focal finding that is real spans adjacent slices;
        # a single-slice spike is usually noise, and the LSE is exactly the
        # pool that would reward it.
        es = e.permute(0, 1, 3, 2).reshape(B * S * n_out, 1, Z)
        es = F.avg_pool1d(F.pad(es, (1, 1), mode="replicate"), 3, stride=1)
        es = es.reshape(B, S, n_out, Z).permute(0, 1, 3, 2)
        tau = self.ev_tau.clamp(min=0.2)                         # (F,)
        lse = tau * (torch.logsumexp(es / tau, dim=2)
                     - math.log(float(Z)))                       # (B,S,F)
        di = self.ev_dif_idx
        ed = e[..., di]                                          # (B,S,Z,Fd)
        w = torch.softmax(self.ev_mix, dim=-1)                   # (Fd,3)
        mix = (w[:, 0] * ed.mean(2)
               + w[:, 1] * torch.quantile(ed, 0.8, dim=2)
               + w[:, 2] * ed.max(2).values)                     # (B,S,Fd)
        ep = lse.clone()
        ep[..., di] = mix                                        # (B,S,F)
        if self.ev2_mode != "off":
            # LEVEL: the residual is added to the PER-SERIES evidence, i.e. at
            # exactly the level of V1's focal LSE, before the reliability
            # weighting and before osseous_factorize. Because sum_s r = 1 this
            # is the study-logit residual of the PREREG, expressed per series so
            # (a) the series mask and the reliability prior apply to it as they
            # do to V1's evidence, and (b) V1's invariant "the broadcast aux BCE
            # supervises exactly the quantity that is summed into the study
            # logit" is preserved (ep IS out[1]). DEVIATION NOTE: the aux BCE
            # therefore also sees the residual; at gamma=0 both choices are
            # bit-identical, so the equivalence gate is unaffected.
            ep = self._ev2_residual(tokens, es, tau, ep, mask, B, S, Z)
        m = mask.to(ep.dtype).unsqueeze(-1)                      # (B,S,1)
        rl = self.ev_R.t()[stype]                                # (B,S,F)
        rl = rl.masked_fill(m == 0, -1e4)
        r = torch.softmax(rl, dim=1)                             # (B,S,F)
        logits = self.ev_b + (r * ep).sum(1)                     # (B,F)
        if self.syno_spec:
            # STUDY-level: the Synovitis residual is added AFTER the reliability
            # sum, not per series, because two of its three branches (B's
            # reliability-weighted burden vector, D's stopgrad Effusion logit)
            # are study-level objects by construction. ep (= out[1], what the
            # trainer's broadcast aux BCE supervises) is therefore UNTOUCHED, and
            # so is every one of the other eleven study logits.
            logits = self._syno_residual(logits, e, es, tau, tokens, r,
                                         mask, B, S, Z)
        # --- evdrop: TRAIN-ONLY masked view (scratch/evdrop/PREREG.md) -----
        # Nothing above this line changes. The full view IS the returned
        # logits; the masked view is a SECOND pool off the SAME tokens (so the
        # trunk is forwarded once) whose focal LSE excludes the single
        # top-evidence slice of each (study, series, focal finding). ev2 and
        # syno_spec are refused in the ctor, so `ep` here is exactly the LSE /
        # mixture pool and the study logit is exactly ev_b + sum_s r*ep.
        if self.evdrop:
            self._evdrop_logits = None
            if (self.training and float(torch.rand(())) < self.evdrop_p
                    and int(Z) >= 2):
                epd = self._evdrop_pool(es, ep, tau, Z)
                self._evdrop_logits = self.ev_b + (r * epd).sum(1)
        # --- rubricrung: TRAIN-ONLY per-rung heads (scratch/rubricrung_20261010)
        # Absent unless enable_rung_heads() was called (train.py --rung-heads),
        # so every existing model / checkpoint / kernel path is untouched.
        if getattr(self, "rung_src", None) is not None:
            self._rung_logits = self._rung_forward(tokens, r, B, S, Z)
        return logits, ep, r.permute(0, 2, 1)

    def enable_rung_heads(self, src_cols):
        """rubricrung (scratch/rubricrung_20261010/DESIGN.md). Adds one extra
        per-slice evidence row per entry of src_cols (finding index f, one row
        per rung), initialised as an exact copy of ev_head row f and ev_b[f].
        The rung row reuses f's OWN tau / LSE-or-mixture pool / reliability
        ev_R[:, f], so a rung row can be swapped into row f of a standard
        EV-MIL checkpoint (rung_surgery.py) and served by the unchanged kernel.
        Refused on every evidence variant whose pool is not exactly
        ev_b + sum_s r * pool(e)."""
        if not getattr(self, "evidence_mil", False):
            raise ValueError("rung heads need --evidence-mil")
        if (getattr(self, "ev2_mode", "off") != "off" or self.syno_spec
                or self.darc_local or self.ev_adapter_mod is not None):
            raise ValueError("rung heads refuse ev2/syno_spec/darc/ev_adapter")
        src = [int(f) for f in src_cols]
        di = [int(x) for x in self.ev_dif_idx.tolist()]
        lin = self.ev_head[2]
        dev = lin.weight.device
        self.rung_src = src
        self.rung_mixrow = [di.index(f) if f in di else -1 for f in src]
        self.rung_w = nn.Parameter(lin.weight.detach()[src].clone())
        self.rung_bias = nn.Parameter(lin.bias.detach()[src].clone())
        self.rung_b = nn.Parameter(self.ev_b.detach()[src].clone())
        self._rung_logits = None
        return self.rung_w.to(dev)

    def _rung_forward(self, tokens, r, B, S, Z):
        """Same pool as _evidence_forward, on the rung rows. r: (B,S,F)."""
        src = self.rung_src
        R = len(src)
        h = self.ev_head[1](self.ev_head[0](tokens))
        e = F.linear(h, self.rung_w, self.rung_bias).reshape(B, S, Z, R).float()
        es = e.permute(0, 1, 3, 2).reshape(B * S * R, 1, Z)
        es = F.avg_pool1d(F.pad(es, (1, 1), mode="replicate"), 3, stride=1)
        es = es.reshape(B, S, R, Z).permute(0, 1, 3, 2)
        tau = self.ev_tau.clamp(min=0.2)[src]                    # (R,)
        lse = tau * (torch.logsumexp(es / tau, dim=2) - math.log(float(Z)))
        w = torch.softmax(self.ev_mix, dim=-1)
        cols = []
        for j, mr in enumerate(self.rung_mixrow):
            if mr < 0:
                cols.append(lse[..., j])
            else:
                ej = e[..., j]                                   # (B,S,Z)
                cols.append(w[mr, 0] * ej.mean(2)
                            + w[mr, 1] * torch.quantile(ej, 0.8, dim=2)
                            + w[mr, 2] * ej.max(2).values)
        ep = torch.stack(cols, dim=-1)                           # (B,S,R)
        return self.rung_b + (r[..., src] * ep).sum(1)

    def _evdrop_pool(self, es, ep, tau, Z):
        """Per-series evidence with the top slice of each focal finding gone.

        es (B,S,Z,F) is the 3-tap SMOOTHED evidence the focal LSE reads; the
        argmax over Z is taken under no_grad (the collaborator's "detach the
        current top-evidence slice": the SELECTION carries no gradient, the
        surviving slices do). The LSE normaliser becomes log(Z-1) so the
        masked view is the same soft-mean statistic over one fewer slice
        rather than a uniformly depressed copy. Diffuse columns are returned
        untouched, bit-for-bit.
        """
        if int(Z) < 2:
            # Nothing to drop: removing the only slice of a series would leave
            # an empty LSE (-inf) and log(Z-1) = log(0) = -inf, i.e. a NaN by
            # construction. A <2-slice series has no "secondary cue" to learn,
            # so the masked view is the full view. Unreachable at the shipped
            # contract (n_slices=24) and guarded at the call site as well.
            return ep
        fo = self.ev_focal                                       # (F,) bool
        esf = es[..., fo]                                        # (B,S,Z,Ff)
        with torch.no_grad():
            top = esf.argmax(dim=2, keepdim=True)                # (B,S,1,Ff)
        km = torch.zeros_like(esf, dtype=torch.bool).scatter_(2, top, True)
        tf = tau[fo]
        # Amendment 1 (build fix, pre-number; 60344923 died with loss nan).
        # The -inf fill must be applied AFTER the division by the temperature,
        # never before it. With `esf.masked_fill(km, -inf) / tf` autograd's
        # gradient for the DIVISOR is grad_out * (-esd / tf**2); at a masked
        # cell grad_out is exactly 0 (logsumexp backward gives exp(-inf)=0)
        # and esd is -inf, so the product is 0 * inf = NaN. That NaN lands in
        # ev_tau at the first optimiser step and poisons every logit from the
        # next forward on. Masking the ALREADY-SCALED evidence keeps -inf out
        # of every differentiable binary op: the forward values are identical
        # (-inf / tf == -inf for tf > 0) and the surviving gradients are the
        # same finite numbers.
        esd = (esf / tf).masked_fill(km, float("-inf"))
        lsed = tf * (torch.logsumexp(esd, dim=2)
                     - math.log(float(Z - 1)))                   # (B,S,Ff)
        epd = ep.clone()
        epd[..., fo] = lsed.to(epd.dtype)
        return epd

    def _ev2_residual(self, tokens, es, tau, ep, mask, B, S, Z):
        """32d morphology residual at V1's evidence locations -> (B,S,F).

        tokens (B*S, Z, D) are the slice-transformer outputs the V1 evidence
        head reads; es is V1's 3-tap SMOOTHED evidence and tau V1's clamped
        temperature, both used DETACHED so the residual cannot move V1's
        localisation (V1's tau still trains through V1's LSE only).
        """
        fi = self.ev2_foc_idx
        if self.ev2_mode == "evw":
            alpha = torch.softmax((es[..., fi] / tau[fi]).detach(), dim=2)
        else:
            # CONTROL: uniform over the SAME slices V1 pools over (V1's focal
            # LSE covers all Z slices of a series; there is no per-slice mask in
            # V1, only the per-series mask, which is applied below).
            alpha = es.new_full((B, S, Z, int(fi.numel())), 1.0 / float(Z))
        h = tokens.reshape(B, S, Z, -1)
        res = []
        for j in range(int(fi.numel())):
            z = self.ev2_proj[j](h)                              # (B,S,Z,32)
            m_f = (alpha[..., j:j + 1].to(z.dtype) * z).sum(2)   # (B,S,32)
            res.append(self.ev2_head[j](m_f).float())            # (B,S,1)
        # Same series mask V1 uses: a padded series contributes nothing (its
        # reliability weight is ~0 anyway, but an explicit zero keeps the
        # residual from depending on padded pixels at all).
        r2 = torch.cat(res, -1) * mask.to(ep.dtype).unsqueeze(-1)  # (B,S,nf)
        ep = ep.clone()
        ep[..., fi] = ep[..., fi] + self.ev2_gamma * r2
        return ep

    def _build_syno_specialist(self, dim):
        """Bundled Synovitis specialist head (scratch/synov2 Amendment 1).

        L_syno = L_V1 + gB.r_B + gC.r_C + gD.r_D. Parameter names start with
        "syno_" and do NOT start with "backbone.", so src/train.py's optimiser
        split puts them in the HEAD LR group, not BBLR - the ev2_ contract.
        gamma init is 0.01, NOT 0: the V2a T4 dynamics probe measured the
        CLAUDE.md "zero-init gates collapse" failure at 0.
        """
        from data import FINDINGS
        self.register_buffer("syno_idx",
                             torch.tensor(int(FINDINGS.index("Synovitis"))),
                             persistent=False)
        self.register_buffer("syno_eff_idx",
                             torch.tensor(int(FINDINGS.index("Effusion"))),
                             persistent=False)
        # B: 9 burden/extent statistics of the per-slice Synovitis evidence.
        self.syno_bmlp = nn.Sequential(nn.Linear(9, 16), nn.GELU(),
                                       nn.Linear(16, 1))
        # C: V2a's morphology machinery on the Synovitis row only.
        self.syno_cproj = nn.Sequential(nn.Linear(dim, 32), nn.LayerNorm(32),
                                        nn.GELU())
        self.syno_cmlp = nn.Sequential(nn.Linear(32, 16), nn.GELU(),
                                       nn.Linear(16, 1))
        # D: stopgrad(V1 study Effusion logit).
        self.syno_dmlp = nn.Sequential(nn.Linear(1, 8), nn.GELU(),
                                       nn.Linear(8, 1))
        self.syno_gamma = nn.Parameter(
            torch.full((3,), float(self.syno_gamma_init)))   # [gB, gC, gD]

    @staticmethod
    def _syno_longest_run(b):
        """(B,S,Z) 0/1 -> (B,S) longest consecutive run, normalised by Z.

        Z = 24, so the scan is 24 cheap elementwise ops - and it is called on
        DETACHED input (a run length has no useful gradient anyway).
        """
        Z = b.shape[-1]
        cur = torch.zeros_like(b[..., 0])
        out = torch.zeros_like(cur)
        for k in range(Z):
            cur = (cur + b[..., k]) * b[..., k]
            out = torch.maximum(out, cur)
        return out / float(Z)

    def _syno_residual(self, logits, e, es, tau, tokens, r, mask, B, S, Z):
        """L_syno += gB.r_B + gC.r_C + gD.r_D. Returns the full (B,F) logits.

        e   (B,S,Z,F) V1's RAW per-slice evidence   -> branch B
        es  (B,S,Z,F) V1's 3-tap SMOOTHED evidence  -> branch C's alpha (detached)
        r   (B,S,F)   V1's reliability weights (softmax over series, sums to 1);
                      used DETACHED so the specialist cannot move V1's series
                      weighting, exactly as V2a detaches alpha.
        """
        ab = str(getattr(self, "syno_ablate", "none") or "none")
        if ab not in ("none", "B", "C", "D", "all"):
            raise ValueError(f"syno_ablate must be none|B|C|D|all, got {ab!r}")
        si = int(self.syno_idx)
        m = mask.to(logits.dtype).unsqueeze(-1)                  # (B,S,1)
        rs = (r[..., si].detach().to(logits.dtype).unsqueeze(-1)
              * m)                                              # (B,S,1)
        add = logits.new_zeros(logits.shape[0], 1)
        if ab not in ("B", "all"):
            p = torch.sigmoid(e[..., si])                        # (B,S,Z)
            stats = [p.mean(2),
                     torch.quantile(p, 0.50, dim=2),
                     torch.quantile(p, 0.80, dim=2),
                     torch.quantile(p, 0.95, dim=2),
                     p.max(2).values,
                     torch.sigmoid((p - 0.50) / 0.05).mean(2),
                     torch.sigmoid((p - 0.75) / 0.05).mean(2),
                     p.std(2),
                     self._syno_longest_run(
                         (p.detach() > 0.5).to(p.dtype))]
            fB = torch.stack(stats, -1).to(logits.dtype)         # (B,S,9)
            add = add + self.syno_gamma[0] * self.syno_bmlp((rs * fB).sum(1))
        if ab not in ("C", "all"):
            alpha = torch.softmax((es[..., si] / tau[si]).detach(), dim=2)
            h = tokens.reshape(B, S, Z, -1)
            z = self.syno_cproj(h)                               # (B,S,Z,32)
            mc = (alpha.unsqueeze(-1).to(z.dtype) * z).sum(2)    # (B,S,32)
            vc = (rs * mc.to(logits.dtype)).sum(1)               # (B,32)
            add = add + self.syno_gamma[1] * self.syno_cmlp(vc)
        if ab not in ("D", "all"):
            eff = logits[:, int(self.syno_eff_idx)].detach().unsqueeze(-1)
            add = add + self.syno_gamma[2] * self.syno_dmlp(eff)
        out = logits.clone()
        out[:, si] = out[:, si] + add.squeeze(-1).to(out.dtype)
        return out

    def _stack_ctx(self, v):
        """(B*S, Z, H, W) -> (B*S*Z, C, H, W) with neighbouring slices as channels."""
        B, Z, H, W = v.shape
        k = self.slice_ctx
        pad = k // 2
        vp = F.pad(v.unsqueeze(1), (0, 0, 0, 0, pad, pad), mode="replicate").squeeze(1)
        chans = [vp[:, i:i + Z] for i in range(k)]
        x = torch.stack(chans, dim=2)           # (B*S, Z, k, H, W)
        return x.reshape(B * Z, k, H, W)

    def enable_patch_export(self, stage=2):
        """Export the backbone's SPATIAL feature map, for scratch/xplanegeom.

        Read-only instrumentation: registers a forward hook on one backbone
        stage and stashes its (N,C,h,w) output in self._patch_feat. Changes no
        weight, no training path, and no forward result - a checkpoint run with
        this enabled produces bit-identical predictions to one without.

        RESOLUTION, and it is load-bearing for the geometry diagnostic. At 384px
        ConvNeXt stages are /4 /8 /16 /32:

        The FOV is ~150 mm across the 384 px input (this file already records
        "at 384px the ConvNeXt map is 12x12, one cell ~= 12.5 mm"), so:

            stage 0   96x96   ~1.6 mm/cell
            stage 1   48x48   ~3.1 mm/cell
            stage 2   24x24   ~6.3 mm/cell
            stage 3   12x12   ~12.5 mm/cell  <- what the pooled path sees

        xplanegate's GAP verdict was measured at a 5 mm tolerance. The FINAL
        stage cannot resolve that - one cell spans ~12.5 mm - so the coarse
        retrieval result was partly a resolution artifact, not purely an absence
        of correspondence. Only stages 0-1 (1.6 / 3.1 mm) are fine enough to
        test 5 mm honestly; stage 2 at 6.3 mm sits right at the boundary.
        The diagnostic must SWEEP stage, not assume one.
        """
        stages = getattr(self.backbone, "stages", None)
        if stages is None:
            raise ValueError(f"{type(self.backbone).__name__} has no .stages")
        if not 0 <= stage < len(stages):
            raise ValueError(f"stage {stage} but backbone has {len(stages)}")
        # REMOVE any hook from a previous call before registering. Without
        # this the hooks ACCUMULATE and every registered stage fires on each
        # forward, so `_patch_feat` ends up holding whichever stage runs LAST
        # (the deepest) regardless of which was requested. Measured: after
        # 3 studies x 4 stages, 12 hooks live and a request for stage 1 (48x48)
        # returned stage 3 (12x12). That fails SILENTLY - the run completes and
        # every stage row is secretly the final stage.
        for h in getattr(self, "_patch_handles", []):
            h.remove()
        self._patch_feat = None
        self._patch_stage = stage
        self._patch_handles = [stages[stage].register_forward_hook(
            lambda m, i, o: setattr(self, "_patch_feat", o.detach()))]
        return self

    def _pool(self, fm):
        """(N,C,H,W) -> (N,D), through the backbone's own head.

        Feeding a (N,C,1,1) map to forward_head is what keeps this backbone-
        agnostic: the head's global pool over a 1x1 map is the identity, so its
        norm/flatten/drop still run exactly as they do in the control. At p=1
        we hand it the untouched map instead, so that path is bit-identical.
        """
        if self.gem_p == 1.0:
            return self.backbone.forward_head(fm, pre_logits=True)
        p = self.gem_p
        s = fm.sign() * fm.abs().clamp(min=1e-6).pow(p)
        m = s.mean((-2, -1), keepdim=True)
        g = m.sign() * m.abs().clamp(min=1e-12).pow(1.0 / p)
        return self.backbone.forward_head(g, pre_logits=True)

    def forward(self, vol, stype, mask, seq=None):
        if self.bn_valid_only and (self.cell_k or self.band_proj is not None
                                   or self.want_seg or self.gem_p != 1.0
                                   or self.pfrel_net is not None
                                   or self.bone_net is not None):
            raise NotImplementedError(
                "--bn-valid-only is wired only on the plain gem_p==1 backbone "
                "path (scratch/truebatch); refusing rather than silently "
                "skipping it on a feature-map branch.")
        # vol: (B, S, Z, H, W)
        if self.want_seg and (self.cell_k or self.band_proj is not None):
            raise NotImplementedError(
                "--aux-seg-w has not been combined with --cell-attn/"
                "--band-pool; not exercised by any launch, fails loudly "
                "rather than silently skipping the decoder.")
        if vol.dim() == 6:
            # --native-ctx loader: (B, S, Z, k, H, W) already carries the
            # context axis (native neighbours); skip _stack_ctx.
            B, S, Z, K, H, W = vol.shape
            if self.xfuse:
                raise ValueError("--xfuse does not accept a native-ctx 6-dim input")
            if K != self.slice_ctx:
                raise ValueError(f"native-ctx input has {K} channels, model "
                                 f"slice_ctx={self.slice_ctx}")
            for _m in self._slice3d_mods:       # empty unless --slice3d != none
                _m.S = Z
            x = vol.reshape(B * S * Z, K, H, W)
        else:
            B, S, Z, H, W = vol.shape
            for _m in self._slice3d_mods:       # empty unless --slice3d != none
                _m.S = Z
            if self.stack_depth:
                # scratch/stackdepth: the DEPTH axis becomes the backbone's
                # channel axis, so through-plane mixing happens inside the
                # convolutions from layer one instead of in a later pooling
                # stage. One backbone forward per SERIES, not per slice; the
                # slice transformer then sees a length-1 sequence and is inert.
                # The study transformer over <= n_series is unchanged.
                if Z != self.slice_ctx:
                    raise ValueError(
                        f"--stack-depth requires slice_ctx ({self.slice_ctx}) "
                        f"== n_slices ({Z}); the depth axis IS the channel axis.")
                x = vol.reshape(B * S, Z, H, W)
                Z = 1
            elif self.xfuse:
                if Z % 2:
                    raise ValueError(f"--xfuse input has odd depth {Z}")
                Z = Z // 2
                v = vol.reshape(B * S, 2 * Z, H, W)
                xa = self._stack_ctx(v[:, :Z].contiguous())      # (N, k, H, W)
                xb = self._stack_ctx(v[:, Z:].contiguous())
                # present = partner channel not identically zero (the loader
                # zero-fills absent partners exactly, after augmentation).
                _pres = (xb.flatten(2).abs().amax(-1) > 0)       # (N, k)
                xa = (xa - self.img_mean) / self.img_std
                xb = ((xb - self.img_mean) / self.img_std) * _pres[:, :, None, None].to(xb.dtype)
                x = torch.cat([xa, xb], 1)                       # (N, 2k, H, W)
                _xf_flag = _pres[:, self.slice_ctx // 2].long()  # (N,)
            else:
                v = vol.reshape(B * S, Z, H, W)
                x = self._stack_ctx(v)              # (B*S*Z, k, H, W)
        if not self.xfuse:
            x = (x - self.img_mean) / self.img_std  # match pretrained input stats
        if self.seq_adapt:
            # rows of x are (b, s, z) in reshape order; one type per series.
            _rt = stype.reshape(B * S).long()
            if bool(((_rt < 0) | (_rt >= 6)).any()):
                raise ValueError("--seq-adapt: series type outside [0, 6)")
            self._sa_ctx["rows"] = _rt.repeat_interleave(x.shape[0] // (B * S))
        if self.pfrel_net is not None:
            # spec s2-s8. ONE axial fluid-sensitive series (S == 1, enforced by
            # the loader's pfrel_series selection), native +/-1 context, no
            # global pool, no slice transformer, no study transformer.
            if S != 1:
                raise ValueError(f"--pfrel serves exactly one series; got S={S}")
            fm = self.backbone.forward_features(x)
            if fm.ndim != 4:
                raise ValueError(
                    f"--pfrel needs a (N,C,H,W) feature map; "
                    f"{type(self.backbone).__name__} returned {tuple(fm.shape)}.")
            _pf = self.pfrel_net(fm, B, Z)                     # (B,)
            n_sc = self.head.shape[0]
            # Exportable as a 12-column predictor: the PF OA column carries the
            # logit, the other eleven are a CONSTANT zero (AUC 0.5 by
            # construction, never a silent number that looks trained). The
            # eleven are muted to weight 0 in the loss by --mute-after, so they
            # receive no gradient either.
            _cols = [_pf if j == PF_OA_IDX else _pf.new_zeros(B)
                     for j in range(n_sc)]
            logits = torch.stack(_cols, dim=1)                 # (B, n_out)
            aux_logits = logits.detach().unsqueeze(1).expand(B, S, n_sc)
            attn = logits.new_zeros(B, n_sc, S)
            return (logits, aux_logits, attn)
        if self.bone_net is not None:
            # Bone-WSRPN V0. <= 2 series (sag FS + cor FS, chosen
            # deterministically by the loader's bone_series mode), native +/-1
            # context, no global pool, no slice/study transformer. Both arms
            # end in the PRODUCTION osseous factorisation.
            if S > 2:
                raise ValueError(f"--bonewsrpn serves <= 2 series; got S={S}")
            self._bone_c3 = None
            fm4 = self.backbone.forward_features(x)
            fm3 = self._bone_c3
            self._bone_c3 = None
            if fm4.ndim != 4 or fm3 is None or fm3.ndim != 4:
                raise ValueError(
                    "--bonewsrpn needs 4-D stage-3 and stage-4 maps; got "
                    f"{None if fm3 is None else tuple(fm3.shape)} / "
                    f"{tuple(fm4.shape)}")
            _zE, _zL = self.bone_net(fm3, fm4, B, S, Z,
                                     stype.to(torch.long), mask)
            n_sc = self.head.shape[0]
            # Exportable as a 12-column predictor: rows 10/11 carry (z_E, z_L)
            # and osseous_factorize turns them into the Contusion/Fracture
            # logits every downstream consumer already expects. The other ten
            # columns are a CONSTANT zero (AUC 0.5 by construction) and are
            # muted to weight 0 in the loss by --mute-after.
            _cols = [_zE if j == CONT_IDX else (_zL if j == FRAC_IDX
                                                else _zE.new_zeros(B))
                     for j in range(n_sc)]
            logits = osseous_factorize(torch.stack(_cols, dim=1))
            aux_logits = logits.detach().unsqueeze(1).expand(B, S, n_sc)
            attn = logits.new_zeros(B, n_sc, S)
            return (logits, aux_logits, attn)
        cells = None
        if self.cell_k:
            # Keep the spatial map instead of letting the backbone average it
            # away. forward_head(pre_logits=True) reproduces backbone(x) EXACTLY,
            # so the series/slice path downstream is bit-identical to the control
            # and only the extra cell branch is new.
            fm = self.backbone.forward_features(x)
            if fm.ndim != 4:
                raise ValueError(
                    f"--cell-attn needs a (N,C,H,W) feature map; "
                    f"{type(self.backbone).__name__} returned {tuple(fm.shape)}.")
            f = self._pool(fm)
            cm = fm if self.cell_stage < 0 else self._cell_feat
            if cm is None:
                raise RuntimeError("--cell-stage hook did not fire; the "
                                   "backbone's forward_features must run the "
                                   "hooked stage")
            self._cell_feat = None
            cells = F.adaptive_avg_pool2d(cm, self.cell_k)   # (N, C, K, K)
            cells = cells.flatten(2).transpose(1, 2)          # (N, K*K, C)
            cells = self.cell_proj(cells)                     # -> (N, K*K, D)
        elif self.band_proj is None:
            if self.want_seg and self.aux_seg_decoder is not None:
                # --aux-seg-w: needs the spatial feature map too. forward_head
                # (pre_logits=True) reproduces backbone(x) EXACTLY (same
                # guarantee the band-pool branch above already relies on), so
                # the pooled global vector `f` feeding everything downstream
                # is bit-identical to the gem_p==1.0 default path; only the
                # decoder's output (self._seg_logits) is new.
                fm = self.backbone.forward_features(x)
                if fm.ndim != 4:
                    raise ValueError(
                        f"--aux-seg-w needs a (N,C,H,W) feature map; "
                        f"{type(self.backbone).__name__} returned {tuple(fm.shape)}.")
                f = self.backbone.forward_head(fm, pre_logits=True)
                self._seg_logits = self.aux_seg_decoder(fm)
            elif self.gem_p == 1.0 and self.bn_valid_only:
                # --bn-valid-only (scratch/truebatch): padded series never
                # reach the trunk. Rows are (b, s, z) in reshape order.
                _zz = x.shape[0] // (B * S)
                _keep = (mask.reshape(B * S) != 0).repeat_interleave(_zz)
                _fk = self.backbone(x[_keep])
                f = _fk.new_zeros(x.shape[0], _fk.shape[1])
                f[_keep] = _fk
            elif self.gem_p == 1.0:
                f = self.backbone(x)            # (B*S*Z, D)  bit-identical path
            else:
                fm = self.backbone.forward_features(x)
                if fm.ndim != 4:
                    raise ValueError(
                        f"--gem-pool needs a (N,C,H,W) feature map; "
                        f"{type(self.backbone).__name__} returned {tuple(fm.shape)}.")
                f = self._pool(fm)
        else:
            fm = self.backbone.forward_features(x)
            if fm.ndim != 4:
                raise ValueError(
                    f"--band-pool needs a (N,C,H,W) feature map; {type(self.backbone).__name__} "
                    f"returned {tuple(fm.shape)}. Token-sequence backbones need a reshape first.")
            # forward_head with pre_logits gives EXACTLY what self.backbone(x)
            # would have returned, so the global branch is unchanged.
            g = self.backbone.forward_head(fm, pre_logits=True)
            bnd = fm.mean(-2)                   # collapse rows -> (N, C, W)
            w = bnd.shape[-1]
            mid = w // 2
            d = max(1, int(round(self.band_delta * w)))
            f = self.band_proj(torch.cat([
                self.band_norm(bnd[..., :mid + 1].mean(-1)),
                self.band_norm(bnd[..., max(0, mid - d):mid + d + 1].mean(-1)),
                self.band_norm(bnd[..., mid:].mean(-1)),
                g], dim=-1))
        if self.xfuse:
            f = f + self.xfuse_emb(_xf_flag).to(f.dtype)
        f = f.reshape(B * S, Z, self.dim)
        if self.want_trunk:
            # Per-study masked mean over (S, Z) of the backbone slice
            # features - the deepest purely-TRUNK representation, taken
            # BEFORE the series encoder so everything above stays free to
            # keep maturing (frztrunk refuted freezing the readout's inputs
            # AND the readout together). mask==1 is a VALID series; a padded
            # series contributes neither to the sum nor to the divisor.
            _fr = f.reshape(B, S, Z, self.dim)
            _m = mask.reshape(B, S, 1, 1).to(_fr.dtype)
            self._trunk_rep = ((_fr * _m).sum((1, 2))
                               / (_m.sum((1, 2)) * Z).clamp(min=1.0))

        if self.ps25:
            _fr = f.reshape(B, S, Z, self.dim)
            _pool = torch.cat([_fr.mean(2), _fr.amax(2)], -1)            # (B,S,2D)
            _st = stype.reshape(B, S).clamp(min=0, max=15).long()
            _slog = self.ps25_head(_pool + self.ps25_type(_st).to(_pool.dtype))   # (B,S,n_out)
            _m = mask.reshape(B, S).to(_slog.dtype)
            _w = _m / _m.sum(1, keepdim=True).clamp(min=1.0)              # mean-fusion weights
            logits = (_slog * _w.unsqueeze(-1)).sum(1)                     # (B,n_out)
            self._study_vec = (_pool * _w.unsqueeze(-1)).sum(1)
            attn = _w.unsqueeze(1).expand(B, logits.shape[-1], S)          # (B,n_out,S)
            return (logits, _slog, attn)

        # stype is flattened to (B*S,) so each series indexes its own slice-
        # position table; position means a different anatomical axis per plane.
        if self.evidence_mil:
            # scratch/evmil: the slice transformer is KEPT (V1); the evidence
            # heads sit on ITS outputs. Everything after it - attn-pooled
            # series token, study transformer, 12 queries, the train-only aux
            # head - is replaced by the constrained evidence sum. The returned
            # per-series evidence IS out[1], so the trainer's existing aux BCE
            # supervises it with the broadcast study target at aux_w
            # (Amendment 2(2): the control's supervision exactly, NOT the
            # auxvis visibility mask).
            series_emb, tokens = self.series_enc(f, stype.reshape(B * S),
                                                 return_tokens=True)
            logits, aux_logits, attn = self._evidence_forward(
                tokens, stype, mask, B, S, Z)
            if self.pat:
                # scratch/shortcutarms §PAT: TRAIN-ONLY, after the full view
                # is complete, so nothing it returns is touched.
                self._pat_logits = None
                if self.training:
                    from pat import patch_logits, pat_aggregate
                    _lp, _pv = patch_logits(self, f, stype, mask, B, S, Z,
                                            self.pat_grid)
                    self._pat_logits = pat_aggregate(_lp, _pv)
            # Kept for the DANN domain head, which lives outside this module
            # and reads a pooled study vector; masked mean over series, the
            # nearest thing this path has to one. Inert in every evmil launch.
            _m = mask.to(series_emb.dtype).reshape(B, S, 1)
            self._study_vec = ((series_emb.reshape(B, S, self.dim) * _m).sum(1)
                               / _m.sum(1).clamp(min=1e-6))
            if self.trauma_head:
                # --trauma-head (scratch/traumahead): non-exclusive pair, REPLACES ossfact on this path.
                _zk, _zks = self._trauma_k(tokens, attn, B, S, Z)
                logits = trauma_decouple(logits, _zk)
                aux_logits = trauma_decouple(aux_logits, _zks)
            elif self.osseous_factor:
                logits = osseous_factorize(logits)
                aux_logits = osseous_factorize(aux_logits)
            if self.oa_gate:
                logits = oa_gate_factorize(logits)
                aux_logits = oa_gate_factorize(aux_logits)
            if self._evdrop_logits is not None:
                # The masked view must pass through EXACTLY the same output
                # factorisations as the full view, or the extra loss term
                # would live in a different logit space.
                _ld = self._evdrop_logits
                if self.osseous_factor:
                    _ld = osseous_factorize(_ld)
                if self.oa_gate:
                    _ld = oa_gate_factorize(_ld)
                self._evdrop_logits = _ld
            if self._pat_logits is not None:
                # Same output factorisations as the full view (evdrop rule).
                _lq = self._pat_logits
                if self.osseous_factor:
                    _lq = osseous_factorize(_lq)
                if self.oa_gate:
                    _lq = oa_gate_factorize(_lq)
                self._pat_logits = _lq
            return (logits, aux_logits, attn)
        if self.rubric_pool:
            # scratch/rubpool: the backbone and the slice transformer are the
            # reference recipe verbatim; only what happens ABOVE the slice
            # tokens changes - the attn-pooled series token, the study
            # transformer and the 12 queries are replaced by the fixed
            # rubric-shaped pooling of per-slice per-finding evidence. The aux
            # per-series head is UNCHANGED (still read off the pooled series
            # token, still supervised by the broadcast aux BCE at --aux-w), so
            # the arm differs from its control in the aggregator alone.
            series_emb, tokens = self.series_enc(f, stype.reshape(B * S),
                                                 return_tokens=True)
            aux_logits = self.aux_head(series_emb).reshape(B, S, -1)
            logits, attn = self._rubric_forward(tokens, mask, B, S, Z)
            # Pooled study vector for the DANN domain head, which lives outside
            # this module; masked mean over series, the nearest thing this path
            # has to one. Inert in every rubpool launch.
            _m = mask.to(series_emb.dtype).reshape(B, S, 1)
            self._study_vec = ((series_emb.reshape(B, S, self.dim) * _m).sum(1)
                               / _m.sum(1).clamp(min=1e-6))
            if self.osseous_factor:
                logits = osseous_factorize(logits)
                aux_logits = osseous_factorize(aux_logits)
            if self.oa_gate:
                logits = oa_gate_factorize(logits)
                aux_logits = oa_gate_factorize(aux_logits)
            return (logits, aux_logits, attn)
        if self.flat_attn:
            series_emb, tokens = self.series_enc(f, stype.reshape(B * S),
                                                 return_tokens=True)
            tokens = tokens.reshape(B, S, Z, self.dim)
        else:
            series_emb = self.series_enc(f, stype.reshape(B * S))   # (B*S, D)
            tokens = None
        aux_logits = self.aux_head(series_emb).reshape(B, S, -1)

        series_emb = series_emb.reshape(B, S, self.dim)
        if cells is not None:
            cells = cells.reshape(B, S, Z, self.cell_k ** 2, self.dim)
        study_emb, attn = self.study_enc(series_emb, stype, mask, seq,
                                         tokens=tokens, cells=cells)  # (B, n_out, D)
        # Pooled study vector, stashed for training-side consumers (the DANN
        # domain head lives OUTSIDE this module so checkpoints stay clean).
        # One mean over a small tensor; inert at inference.
        # Split the scored queries from the auxiliary ones. Everything
        # downstream that pools "the study" must see only the scored twelve, or
        # the text head and the DANN domain head silently change meaning the
        # moment --report-aux is switched on.
        n_sc = self.head.shape[0]
        aux_emb = study_emb[:, n_sc:] if study_emb.shape[1] > n_sc else None
        study_emb = study_emb[:, :n_sc]
        self._study_vec = study_emb.mean(1)
        if self.mmix_p > 0 and self.training:   # regbundle: stash UNMIXED z
            self._mmix_z = study_emb
        if self.dismix_on and self.training:   # dismix: stash UNMIXED z
            self._dismix_z = study_emb
        if self.pcol_on and self.training:      # recipe2: stash pre-head z
            self._pcol_z = study_emb
        logits = (self.drop(study_emb) * self.head).sum(-1) + self.head_b
        if self.osseous_factor:
            logits = osseous_factorize(logits)
            aux_logits = osseous_factorize(aux_logits)
        if self.oa_gate:
            logits = oa_gate_factorize(logits)
            aux_logits = oa_gate_factorize(aux_logits)
        # Extra outputs are appended in a FIXED order - txt then report_aux -
        # and read from the END by the trainer. Inserting anywhere else would
        # silently shift out[3], which is how a 4-tuple unpack of a 5-tuple
        # already cost half a submission once.
        out = [logits, aux_logits, attn]
        if self.text_head is not None:
            # Pool the per-finding vectors back to one study vector for the
            # text objective; the findings share a study, not a sentence.
            out.append(F.normalize(self.text_head(study_emb.mean(1)), dim=-1))
        if self.aux_w is not None and aux_emb is not None:
            out.append((self.drop(aux_emb) * self.aux_w).sum(-1) + self.aux_b)
        # MOAKS subregional aux logits, appended LAST like report_aux (the two
        # are mutually exclusive, so out[-1] is unambiguous either way). One
        # auxiliary query per block; block logits concatenated in the frozen
        # data.MOAKS_AUX_COLS order -> (B, 62).
        if self.moaks_heads is not None and aux_emb is not None:
            out.append(torch.cat(
                [h(self.drop(aux_emb[:, bi]))
                 for bi, h in enumerate(self.moaks_heads)], dim=-1))
        return tuple(out)

    def mmix_logits(self, cell_ok):
        """Manifold mixup step (train only; call ONCE per training study,
        right after forward, with mmix_p > 0). `cell_ok` is a (B, n_out)
        bool: cells where BOTH the current study and the banked one carry a
        supervised target. With prob mmix_p (and a bank of matching shape),
        draws lam ~ Beta(alpha, alpha) and returns
            (logits of z_mix, lam_cell)   where z_mix = lam_cell*z + (1-lam_cell)*z_bank
        with lam_cell = lam on ok cells and 1.0 (unmixed) elsewhere, so the
        trainer mixes y / weight with the SAME per-cell lam. Otherwise
        returns (None, None). Always replaces the bank with the current
        (detached) z afterwards, so bank and trainer-side target bank stay
        in lock-step. Only the primary 12-finding logits are recomputed;
        every other output of forward() (aux per-series, text, report-aux,
        MOAKS, _study_vec) already used the UNMIXED z."""
        z = self._mmix_z
        if z is None:
            raise RuntimeError("mmix_logits: no stashed z (call after a "
                               "training-mode forward with mmix_p > 0)")
        self._mmix_z = None
        bank = self._mmix_bank
        logits, lam_cell = None, None
        if (bank is not None and bank.shape == z.shape
                and self._mmix_rng.random() < self.mmix_p):
            lam = self._mmix_rng.betavariate(self.mmix_alpha, self.mmix_alpha)
            lam_cell = torch.where(cell_ok.to(z.device),
                                   torch.full(cell_ok.shape, lam, device=z.device),
                                   torch.ones(cell_ok.shape, device=z.device))
            lc = lam_cell.to(z.dtype)[..., None]
            zm = lc * z + (1.0 - lc) * bank.to(z.dtype)
            logits = (self.drop(zm) * self.head).sum(-1) + self.head_b
            if self.osseous_factor:
                logits = osseous_factorize(logits)
            if self.oa_gate:
                logits = oa_gate_factorize(logits)
        self._mmix_bank = z.detach()
        return logits, lam_cell


def osseous_factorize(z, eps=1e-6):
    """Factorize the trauma pair: Contusion (10) and Fracture (11) share one
    axis by RUBRIC CONSTRUCTION - Fracture is impact marrow oedema WITH a
    discrete cortical line, Contusion the same oedema WITHOUT one. Two
    independent sigmoids make the oedema-positive non-fractures label noise
    that both heads fight over; measured across 80 arms, the Contusion head
    scores the FRACTURE label better than the Fracture head does (0.8741 vs
    0.8662) and the dedicated head contributes -0.008.

    Reinterpret the two head rows as z_E (oedema present - reads on
    fluid-sensitive series) and z_L (line | oedema - reads on non-FS
    high-res), and emit

        P(Contusion) = P(E) * (1 - P(L|E))       P(Fracture) = P(E) * P(L|E)

    converted BACK to logits, so every downstream consumer - soft_bce, the
    aux per-series loss, OOF export, the notebook's sigmoid - is untouched.
    The oedema-positive non-fractures thereby become the explicit positive
    signal for 1-P(L|E) instead of noise. Gradient reweighting on the same
    subgroup (--hard-neg) moved the target column BACKWARDS; this changes the
    probability structure, not the gradient allocation, and is untested.

    fp32 inside regardless of autocast; clamped logits stay within +-13.8 so
    the cast back to fp16 is safe.
    """
    e = torch.sigmoid(z[..., 10].float())
    l = torch.sigmoid(z[..., 11].float())
    pc = (e * (1.0 - l)).clamp(eps, 1.0 - eps)
    pf = (e * l).clamp(eps, 1.0 - eps)
    z = z.clone()
    z[..., 10] = (pc.log() - torch.log1p(-pc)).to(z.dtype)
    z[..., 11] = (pf.log() - torch.log1p(-pf)).to(z.dtype)
    return z


def trauma_decouple(z, zk, eps=1e-6):
    """--trauma-head (scratch/traumahead/PREREG.md): NON-EXCLUSIVE trauma pair.
    Rows 10/11 keep the ossfact meaning z_E (oedema present) / z_L (line | oedema); zk is the independent contusion factor:
        P(Contusion) = P(E) * P(K)      P(Fracture) = P(E) * P(L)
    so C and F can both approach E (co-positives) instead of summing to <= E. zk has z's shape minus the last dim.
    Same emit-as-logits contract and fp32/clamp as osseous_factorize."""
    e = torch.sigmoid(z[..., 10].float())
    k = torch.sigmoid(zk.float())
    l = torch.sigmoid(z[..., 11].float())
    pc = (e * k).clamp(eps, 1.0 - eps)
    pf = (e * l).clamp(eps, 1.0 - eps)
    z = z.clone()
    z[..., 10] = (pc.log() - torch.log1p(-pc)).to(z.dtype)
    z[..., 11] = (pf.log() - torch.log1p(-pf)).to(z.dtype)
    return z


def oa_gate_factorize(z, eps=1e-6):
    """Compartment-gated OA triad: rows 4/5/6 (Medial/Lateral/PF OA) become
    P(i) = G * sigma(z_i) with G = sigma(max(z4, z5, z6)) - a shared
    "degenerative disease present" gate taken from the WORST compartment.

    Non-vacuous (each column is capped by the gate, so the three heads share
    the disease-detection axis and specialise on compartment assignment),
    unlike the additive z_common +- z_diff split, which for linear heads is a
    pure change of basis. max (not mean) keeps unilateral disease expressive:
    a medial-only knee needs G high WITHOUT dragging z_lat up. Same emit-as-
    logits contract as osseous_factorize: every consumer downstream is
    untouched. Parameter-free -> read it from ck["args"], never sniff.

    Motivation: Lateral OA (probe 0.908) is the rarer compartment whose head
    gets noisier gradient; the gate lets it inherit the well-estimated shared
    disease signal. Follows the ossfact pattern (rubric/anatomy-structural
    output factorization) - pre-registered before ossfact's own verdict.
    """
    zs = z[..., 4:7].float()
    g = torch.sigmoid(zs.max(-1).values)
    z = z.clone()
    for j in (4, 5, 6):
        p = (g * torch.sigmoid(z[..., j].float())).clamp(eps, 1.0 - eps)
        z[..., j] = (p.log() - torch.log1p(-p)).to(z.dtype)
    return z


def soft_bce(logits, target, weight=None, sharpen=None):
    """BCE against soft targets, since report grades give probabilities.

    `sharpen`, when given, is a per-cell magnitude s >= 0 implementing the
    negative-r half of generalized label smoothing:

        L = (1 + s) * BCE  -  s * KL(p || uniform)

    KL-to-uniform is log2 - H(p), so SUBTRACTING it rewards low entropy: the
    term pushes predictions away from 0.5 rather than toward it. It is an
    anti-regulariser, applied only where the report stated a finding
    categorically, and it is the half of the mechanism our pipeline has never
    had - the calibrated ladder only ever SOFTENS.

    s = 0 is exactly the old loss, cell by cell.
    """
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if sharpen is not None:
        # H(p) via logsigmoid, in fp32, so neither tail underflows at bf16.
        z = logits.float()
        p = torch.sigmoid(z)
        h = -(p * F.logsigmoid(z) + (1.0 - p) * F.logsigmoid(-z))
        kl_unif = math.log(2.0) - h              # 0 at p=0.5, log2 at 0 or 1
        loss = (1.0 + sharpen) * loss - sharpen * kl_unif.to(loss.dtype)
    if weight is not None:
        loss = loss * weight
    return loss.mean()
