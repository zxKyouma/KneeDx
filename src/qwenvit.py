"""Qwen3.5-2B vision tower as a KneeNet slice encoder (scratch/qwenvit/DESIGN in PREREG.md).

--backbone qwen35v_2b : NEW additive name (no pre-existing --backbone value reaches it).

Input : (N, 3, H, W) already ImageNet-standardised by KneeNet ((x - .485/.456/.406) / .229/.224/.225).
        The wrapper undoes that and applies Qwen's own normalisation (x - .5) / .5 (preprocessor_config.json),
        so KneeNet is untouched.
Tower : HF Qwen3_5VisionModel modules (config vendored below, so building needs no hub access), run by a
        BATCHED forward (all images share one grid, so per-image sdpa over (N, heads, T, d) is exactly the
        packed/cu_seqlens computation HF does image-by-image). Patch embed: HF's Conv3d over a temporally
        duplicated still image == conv2d with the kernel summed over its temporal axis. Token order = HF's
        merge-block order (h/2, w/2, 2, 2) so HF's own position-id / bilinear pos-embed helpers apply verbatim.
        Equivalence vs HF Qwen3_5VisionModel(pixel_values from Qwen2VLImageProcessor) is checked in the smoke.
Pool  : merger output (the 2048-d tokens the LLM consumes), mean over the 144 merged tokens of a 384 image.
Adapter (qwenvlm v2 lessons): RunningStd per-dim ACROSS-image standardisation (running stats, data-dependent
        init on the first training forward, momentum .01, frozen in eval; persistent buffers -> in the ckpt and
        EMA-averaged by train.EMA), then Linear(2048 -> 768) + LayerNorm(768) so the slice transformer sees the
        same width/statistics as ConvNeXt-Tiny's pooled+normed output (num_features = 768).
Training-only env (read only when pretrained=True, never at inference / ckpt_build):
    QWENVIT_FREEZE=k  freeze patch_embed + pos_embed + first k blocks (default 0).
    QWEN35_2B_DIR     local HF snapshot dir with model.safetensors* (default: .hf cache snapshot).
Inference env: QWENVIT_CHUNK=n images per tower call under no_grad (default 0 = all at once).
"""
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

VISION_CONFIG = dict(depth=24, hidden_act="gelu_pytorch_tanh", hidden_size=1024, in_channels=3,
                     initializer_range=0.02, intermediate_size=4096, num_heads=16,
                     num_position_embeddings=2304, out_hidden_size=2048, patch_size=16,
                     spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[])
QWEN_DEFAULT_DIR = os.environ.get("RSNA_QWEN_DIR", os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub/models--Qwen--Qwen3.5-2B/snapshots"))
OUT_DIM = 768


class RunningStd(nn.Module):
    def __init__(self, d, mom=0.01):
        super().__init__()
        self.mom = mom
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("var", torch.ones(d))
        self.register_buffer("n_upd", torch.zeros(()))

    def forward(self, x):                                   # x (N, d) float32
        if self.training:
            with torch.no_grad():
                xd = x.detach().float()
                bm = xd.mean(0)
                bv = ((xd - bm) ** 2).mean(0) + 1e-6
                if float(self.n_upd) == 0.0:                # data-dependent init
                    self.mu.copy_(bm); self.var.copy_(bv)
                else:
                    self.mu.mul_(1 - self.mom).add_(bm, alpha=self.mom)
                    self.var.mul_(1 - self.mom).add_(bv + (bm - self.mu) ** 2, alpha=self.mom)
                self.n_upd.add_(1.0)
        return (x.float() - self.mu) / torch.sqrt(self.var + 1e-5)


def _rope(q, cos, sin):                                     # q (N, T, H, d); cos/sin (T, d)
    qf = q.float()
    c, s = cos[None, :, None, :].float(), sin[None, :, None, :].float()
    x1, x2 = qf[..., : qf.shape[-1] // 2], qf[..., qf.shape[-1] // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return (qf * c + rot * s).to(q.dtype)


class QwenVisionSlice(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        cfg = Qwen3_5VisionConfig(**VISION_CONFIG)
        cfg._attn_implementation = "sdpa"
        self.visual = Qwen3_5VisionModel(cfg)
        self.cfg = cfg
        self.stdz = RunningStd(cfg.out_hidden_size)
        self.proj = nn.Linear(cfg.out_hidden_size, OUT_DIM)
        self.norm = nn.LayerNorm(OUT_DIM)
        self.num_features = OUT_DIM
        self.grad_ckpt = False
        self._geo = {}
        self.register_buffer("in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        if pretrained:
            self._load_pretrained()
            k = int(os.environ.get("QWENVIT_FREEZE", "0"))
            if k > 0:
                frz = [self.visual.patch_embed, self.visual.pos_embed] + list(self.visual.blocks[:k])
                n = 0
                for m in frz:
                    for p in m.parameters():
                        p.requires_grad_(False); n += p.numel()
                print(f"[qwenvit] froze patch_embed+pos_embed+blocks[:{k}] ({n/1e6:.1f}M params)", flush=True)

    def _load_pretrained(self):
        from safetensors import safe_open
        d = os.environ.get("QWEN35_2B_DIR", "")
        if not d:
            snaps = sorted(glob.glob(os.path.join(QWEN_DEFAULT_DIR, "*")))
            if not snaps:
                raise SystemExit(f"qwen35v_2b: no snapshot under {QWEN_DEFAULT_DIR} (set QWEN35_2B_DIR)")
            d = snaps[-1]
        sd = {}
        for fn in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            with safe_open(fn, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("model.visual."):
                        sd[k[len("model.visual."):]] = f.get_tensor(k).float()
        missing, unexpected = self.visual.load_state_dict(sd, strict=False)
        if missing or unexpected or not sd:
            raise SystemExit(f"qwen35v_2b: key mismatch missing={missing[:8]} unexpected={unexpected[:8]} n={len(sd)}")
        print(f"[qwenvit] loaded {len(sd)} vision tensors from {d}", flush=True)

    def set_grad_checkpointing(self, enable=True):
        self.grad_ckpt = bool(enable)

    def _geometry(self, gh, gw, device):
        key = (gh, gw, str(device))
        if key not in self._geo:
            from transformers.vision_utils import (get_vision_bilinear_indices_and_weights,
                                                   get_vision_position_ids)
            v = self.visual
            thw = torch.tensor([[1, gh, gw]], device=device)
            bi, bw = get_vision_bilinear_indices_and_weights(thw, num_grid_per_side=v.num_grid_per_side,
                                                             spatial_merge_size=v.spatial_merge_size)
            pid = get_vision_position_ids(thw, v.spatial_merge_size)
            rot = v.rotary_pos_emb(pid.to(device))
            emb = torch.cat((rot, rot), dim=-1)
            self._geo[key] = (bi, bw, emb.cos(), emb.sin())
        return self._geo[key]

    def _block(self, blk, h, cos, sin):
        N, T, C = h.shape
        H = blk.attn.num_heads
        x = blk.norm1(h)
        q, k, v = blk.attn.qkv(x).view(N, T, 3, H, C // H).unbind(2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        h = h + blk.attn.proj(a.transpose(1, 2).reshape(N, T, C))
        return h + blk.mlp(blk.norm2(h))

    def tower(self, x):                                     # x (N,3,H,W) raw [0,1] -> (N, 2048)
        v = self.visual
        P, m = v.patch_size, v.spatial_merge_size
        N, _, Hh, Ww = x.shape
        gh, gw = Hh // P, Ww // P
        xq = (x - 0.5) / 0.5
        w = v.patch_embed.proj.weight.sum(2)                # Conv3d over a duplicated frame == conv2d(sum_t W_t)
        h = F.conv2d(xq.to(w.dtype), w, v.patch_embed.proj.bias, stride=P)      # (N, C, gh, gw)
        C = h.shape[1]
        h = h.view(N, C, gh // m, m, gw // m, m).permute(0, 2, 4, 3, 5, 1).reshape(N, gh * gw, C)
        bi, bw, cos, sin = self._geometry(gh, gw, x.device)
        pos = (v.pos_embed(bi) * bw[:, :, None]).sum(0)     # (T, C) in merge-block order
        h = h + pos.to(h.dtype)[None]
        for blk in v.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                h = checkpoint(self._block, blk, h, cos, sin, use_reentrant=False)
            else:
                h = self._block(blk, h, cos, sin)
        mg = v.merger
        z = mg.norm(h).reshape(N, (gh * gw) // (m * m), -1)
        z = mg.linear_fc2(mg.act_fn(mg.linear_fc1(z)))       # (N, T/4, 2048)
        return z.mean(1)

    def forward(self, x):
        x = x * self.in_std + self.in_mean                  # undo KneeNet's ImageNet standardisation
        ch = int(os.environ.get("QWENVIT_CHUNK", "0"))
        if ch > 0 and not torch.is_grad_enabled() and x.shape[0] > ch:
            f = torch.cat([self.tower(x[i:i + ch]) for i in range(0, x.shape[0], ch)], 0)
        else:
            f = self.tower(x)
        return self.norm(self.proj(self.stdz(f.float())))
