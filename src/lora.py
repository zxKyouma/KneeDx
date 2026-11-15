"""LoRA adapters on a frozen 2D ViT trunk (scratch/dinofam/PREREG.md, "dinolora arm").

Mechanism (AnyMC3D, arXiv 2512.12887, VLM3D 2025 1st place): freeze the
foundation ViT entirely and train only rank-r updates on the q and v
projections of every attention block, plus the downstream slice/study
transformer and heads. No dependency on `peft` (not installed in envs/knee).

Design constraints this module honours
--------------------------------------
* ADDITIVE: nothing here runs unless train.py is given --lora-rank > 0.
* STATE-DICT LAYOUT: the wrapped projection keeps the exact parameter objects
  and key names of the timm Linear it replaces (`...attn.qkv.weight`), and adds
  `...attn.qkv.lora_A_q / lora_B_q / lora_A_v / lora_B_v`. So the trained
  checkpoint is a strict SUPERSET of a plain ViT state dict and EMA / the
  raw<->EMA load_state_dict round-trips in train.py work unchanged.
* CONSUMERS (export_oof / eval_checkpoints / infer / the notebook) build a plain
  timm ViT from ck["args"]["backbone"] and know nothing about LoRA. They read a
  checkpoint produced by `python src/lora.py merge in.pt out.pt`, which folds
  W <- W + (alpha/r) * B @ A into the q and v rows, drops the lora_* keys and
  sets args["lora_rank"] = 0. Merging is exact (linear), so the merged plain
  model reproduces the LoRA model's outputs to float precision; the smoke
  test (scratch/dinofam/lora_smoke.py) asserts this.
* B is zero-initialised (Hu et al. 2021), so at step 0 the LoRA model is
  byte-identical in output to the frozen trunk; the smoke test asserts this too.

Supported attention layouts (timm 1.0.22):
  - fused `attn.qkv` nn.Linear(dim, 3*dim) called as a MODULE (timm
    layers.attention.Attention; models.eva.EvaAttention with q_bias None,
    which is what vit_base_patch16_dinov3 uses: qkv_bias=False).
  - separate `attn.q_proj` / `attn.v_proj` nn.Linear (EvaAttention with
    qkv_fused=False).
EvaAttention with FUSED qkv AND biases calls F.linear on `self.qkv.weight`
directly, bypassing the module forward - that layout is refused loudly
rather than silently training adapters that never run.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_KEYS = ("lora_A_q", "lora_B_q", "lora_A_v", "lora_B_v")


def _init_pair(dim_in, dim_out, rank, ref):
    A = nn.Parameter(torch.empty(rank, dim_in, device=ref.device, dtype=ref.dtype))
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    B = nn.Parameter(torch.zeros(dim_out, rank, device=ref.device, dtype=ref.dtype))
    return A, B


class LoRAQKV(nn.Linear):
    """Fused qkv Linear with rank-r LoRA on the q and v output slices.

    Subclass of nn.Linear sharing the ORIGINAL weight/bias Parameter objects, so
    `state_dict()` keys for the base weights are unchanged.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        dim_in, dim_out = base.in_features, base.out_features
        if dim_out != 3 * dim_in:
            raise ValueError(f"LoRAQKV expects out=3*in, got {dim_in}->{dim_out}")
        super().__init__(dim_in, dim_out, bias=base.bias is not None,
                         device=base.weight.device, dtype=base.weight.dtype)
        self.weight = base.weight          # same Parameter object (pretrained)
        if base.bias is not None:
            self.bias = base.bias
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_A_q, self.lora_B_q = _init_pair(dim_in, dim_in, rank, base.weight)
        self.lora_A_v, self.lora_B_v = _init_pair(dim_in, dim_in, rank, base.weight)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        dq = F.linear(F.linear(x, self.lora_A_q), self.lora_B_q)
        dv = F.linear(F.linear(x, self.lora_A_v), self.lora_B_v)
        delta = torch.cat([dq, torch.zeros_like(dq), dv], dim=-1)
        return out + delta * self.scale


class LoRALinear(nn.Linear):
    """Single projection (q_proj or v_proj) with rank-r LoRA; keys lora_A_q/lora_B_q
    are reused for both so the merge tool needs one code path (the `_q` suffix
    is then just a name)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(base.in_features, base.out_features,
                         bias=base.bias is not None,
                         device=base.weight.device, dtype=base.weight.dtype)
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_A_q, self.lora_B_q = _init_pair(base.in_features, base.out_features,
                                                  rank, base.weight)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return out + F.linear(F.linear(x, self.lora_A_q), self.lora_B_q) * self.scale


def inject_lora(backbone: nn.Module, rank: int, alpha: float):
    """Replace every attention q/v projection in `backbone` with a LoRA-wrapped
    twin. Returns (n_blocks_wrapped, n_lora_params). Raises if nothing matched."""
    if rank <= 0:
        raise ValueError("inject_lora called with rank <= 0")
    n_blocks, n_params = 0, 0
    for name, mod in list(backbone.named_modules()):
        qkv = getattr(mod, "qkv", None)
        q_proj = getattr(mod, "q_proj", None)
        v_proj = getattr(mod, "v_proj", None)
        if isinstance(qkv, nn.Linear):
            if getattr(mod, "q_bias", None) is not None and not getattr(mod, "qkv_bias_separate", False):
                raise SystemExit(f"[lora] {name}: EvaAttention with fused qkv + biases "
                                 f"bypasses qkv.forward (F.linear on qkv.weight); "
                                 f"LoRA would be silently inert. Refusing.")
            new = LoRAQKV(qkv, rank, alpha)
            mod.qkv = new
            n_blocks += 1
            n_params += sum(getattr(new, k).numel() for k in LORA_KEYS)
        elif isinstance(q_proj, nn.Linear) and isinstance(v_proj, nn.Linear):
            mod.q_proj = LoRALinear(q_proj, rank, alpha)
            mod.v_proj = LoRALinear(v_proj, rank, alpha)
            n_blocks += 1
            n_params += (mod.q_proj.lora_A_q.numel() + mod.q_proj.lora_B_q.numel()
                         + mod.v_proj.lora_A_q.numel() + mod.v_proj.lora_B_q.numel())
    if n_blocks == 0:
        raise SystemExit(f"[lora] no attention q/v projection found in "
                         f"{type(backbone).__name__}; LoRA is for ViT-family trunks")
    return n_blocks, n_params


def is_lora_param(name: str) -> bool:
    return ".lora_" in name


def freeze_backbone(model: nn.Module, keep_lora: bool = True):
    """requires_grad=False for every `backbone.` parameter except LoRA ones.
    Returns (n_frozen_tensors, n_frozen_elems, n_trainable_backbone_elems)."""
    nf, ne, nt = 0, 0, 0
    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            continue
        if keep_lora and is_lora_param(n):
            nt += p.numel()
            continue
        p.requires_grad_(False)
        nf += 1
        ne += p.numel()
    return nf, ne, nt


@torch.no_grad()
def merge_state_dict(sd: dict, rank: int, alpha: float) -> dict:
    """Fold LoRA into the base weights and drop the lora_* keys. Pure function on
    a state dict (works for raw, EMA and sub-epoch files alike)."""
    scale = float(alpha) / float(rank)
    out = {k: v.clone() for k, v in sd.items() if ".lora_" not in k}
    prefixes = sorted({k[: k.rfind(".") + 1] for k in sd if ".lora_" in k})
    n_merged = 0
    for pre in prefixes:
        W = out[pre + "weight"]
        Aq, Bq = sd[pre + "lora_A_q"].float(), sd[pre + "lora_B_q"].float()
        dq = (Bq @ Aq) * scale
        if pre + "lora_A_v" in sd:  # fused qkv: q rows, then v rows
            dim = W.shape[1]
            if W.shape[0] != 3 * dim or dq.shape != (dim, dim):
                raise SystemExit(f"[lora-merge] {pre}: shape mismatch {tuple(W.shape)} vs {tuple(dq.shape)}")
            Av, Bv = sd[pre + "lora_A_v"].float(), sd[pre + "lora_B_v"].float()
            dv = (Bv @ Av) * scale
            Wf = W.float()
            Wf[:dim] += dq
            Wf[2 * dim:] += dv
            out[pre + "weight"] = Wf.to(W.dtype)
        else:  # single projection
            if dq.shape != tuple(W.shape):
                raise SystemExit(f"[lora-merge] {pre}: shape mismatch {tuple(W.shape)} vs {tuple(dq.shape)}")
            out[pre + "weight"] = (W.float() + dq).to(W.dtype)
        n_merged += 1
    if n_merged == 0:
        raise SystemExit("[lora-merge] no lora_* keys in this state dict")
    return out


def merge_checkpoint(src_path: str, dst_path: str):
    ck = torch.load(src_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    rank, alpha = int(a.get("lora_rank", 0)), float(a.get("lora_alpha", 0))
    if rank <= 0:
        raise SystemExit(f"{src_path}: args.lora_rank={rank}; not a LoRA checkpoint")
    n_before = len(ck["model"])
    ck["model"] = merge_state_dict(ck["model"], rank, alpha)
    a["lora_merged_from"] = {"src": src_path, "lora_rank": rank, "lora_alpha": alpha}
    a["lora_rank"] = 0          # consumers reading ck["args"] now see a plain trunk
    torch.save(ck, dst_path)
    print(f"[lora-merge] {src_path} -> {dst_path}: {n_before} -> {len(ck['model'])} tensors "
          f"(rank {rank}, alpha {alpha})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "merge":
        raise SystemExit("usage: python src/lora.py merge <in.pt> <out.pt>")
    merge_checkpoint(sys.argv[2], sys.argv[3])
