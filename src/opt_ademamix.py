"""AdEMAMix (Pagliardini, Ablin, Grangier; ICLR 2025, arXiv:2409.03137).

AdamW with a SECOND, very slow EMA of the gradient mixed into the numerator:

    m1 = b1*m1 + (1-b1)*g            fast EMA   (AdamW's momentum; OneCycle
                                                 cycles b1 exactly as for AdamW
                                                 because `betas` is a group key)
    m2 = b3(t)*m2 + (1-b3(t))*g      slow EMA   (NOT bias-corrected, as in the paper)
    v  = b2*v  + (1-b2)*g^2
    theta -= lr * ( (m1/(1-b1^t) + alpha(t)*m2) / (sqrt(v/(1-b2^t)) + eps)
                    + wd*theta )                 decoupled weight decay, per group

Paper schedulers (Alg. 1, eq. 2-3), both warmed over T_warm steps:
    alpha(t) = min(t*alpha/T_warm, alpha)
    b3(t)    = min( exp( ln(b3_start)*ln(b3) /
                         ((1-t/T_warm)*ln(b3) + (t/T_warm)*ln(b3_start)) ), b3 )
with b3_start = b1. The paper uses T_warm = T (full run) for from-scratch
runs; that is what --ademamix-warmup 1.0 gives here.

Horizon scaling (stated for the prereg): the paper's b3 = 0.9999 (time constant
10k steps) sits at ~1-4% of its 256k-1M-step horizons. Our arm has ~5.5k
optimizer steps (10 ep, bs1 accum8, cotrainA pool), so 0.9999 would never fill
the slow EMA. Default b3 = 0.999 (time constant 1k steps ~ 1.8 epochs, ~18% of
the run) keeps the "older gradients" mechanism alive on our horizon; alpha = 5
is the paper's smallest recommended value. Both are flags.

`scaler.unscale_` / `scaler.step` / OneCycleLR / --wd-from all work through the
torch.optim.Optimizer interface (group 'lr', 'betas', 'weight_decay'), so the
rest of the recipe is untouched.
"""
import math

import torch


class AdEMAMix(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta3=0.999,
                 alpha=5.0, warmup_steps=0, eps=1e-8, weight_decay=0.0):
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"beta3 out of range: {beta3}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0: {alpha}")
        defaults = dict(lr=lr, betas=betas, beta3=beta3, alpha=alpha,
                        warmup_steps=int(warmup_steps), eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _sched(t, group):
        """alpha(t), beta3(t) per the paper's linear / log-space warmups."""
        T = group["warmup_steps"]
        alpha, b3 = group["alpha"], group["beta3"]
        b3_start = group["betas"][0]
        if T <= 0 or t >= T:
            return alpha, b3
        f = t / T
        alpha_t = alpha * f
        ln_s, ln_e = math.log(b3_start), math.log(b3)
        b3_t = math.exp(ln_s * ln_e / ((1.0 - f) * ln_e + f * ln_s))
        return alpha_t, min(b3_t, b3)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse grads")
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m1"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["m2"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["step"] += 1
                t = st["step"]
                alpha_t, b3_t = self._sched(t, group)
                m1, m2, v = st["m1"], st["m2"], st["v"]
                m1.mul_(b1).add_(g, alpha=1.0 - b1)
                m2.mul_(b3_t).add_(g, alpha=1.0 - b3_t)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                bc1 = 1.0 - b1 ** t
                bc2 = 1.0 - b2 ** t
                denom = (v / bc2).sqrt_().add_(eps)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                update = m1 / bc1
                if alpha_t > 0:
                    update = update.add(m2, alpha=alpha_t)
                p.addcdiv_(update, denom, value=-lr)
        return loss

    def schedule_at(self, t):
        """Diagnostic: (alpha(t), beta3(t)) for the first group."""
        return self._sched(t, self.param_groups[0])
