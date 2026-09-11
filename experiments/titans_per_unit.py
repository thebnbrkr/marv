"""Does storage localize once units are actually ALLOWED to forget at
different rates -- the thing the real architecture doesn't permit?

titans_ablation.py found no single-unit localization in titans-pytorch's
NeuralMemory. Reading the implementation plus the actual paper
(arXiv:2501.00663) explains why that's close to a structural guarantee, not
a discovery: `to_decay_factor` / `to_adaptive_step` output ONE scalar per
head per chunk, broadcast identically onto all 256 hidden units. Appendix C's
general derivation (eq 32) writes the gate as `diag(1 - alpha_t)`, which DOES
permit a different decay per dimension -- but the released code collapses it
to a scalar (see experiments/README.md for the full writeup).

An earlier version of this script tried to LEARN per-unit decay/lr end to
end (meta-training the controller, mirroring how the real library trains
`to_decay_factor`). That training didn't converge in a reasonable number of
steps -- gradients to the outer controller were real but tiny, likely
because backprop-through-24-sequential-online-updates gives a very indirect,
low-signal path from final loss to controller weights. Rather than spend
more effort debugging that convergence, this version sidesteps it: instead
of LEARNING each unit's decay rate, we ASSIGN it directly and see what
happens. That's a valid, more direct test of the actual question ("does
per-unit decay granularity change whether storage localizes"), independent
of whether a real network would ever learn to set decay rates that way.

Setup, isolating decay as the one changed variable:
- Same 2-layer GELU memory MLP (dim -> hidden -> dim) as titans_pytorch's
  MemoryMLP(dim, depth=2, expansion_factor=4) and titans_ablation.py.
- Raw key = raw value = the pair itself (same autoassociative convention
  used everywhere else in this work) -- no learned key/value/query
  projections, to avoid re-introducing a confound we'd have to untangle.
- Write strength (lr) is the SAME constant for every unit in both
  conditions -- only decay differs between conditions.
- Two decay conditions, same pairs, same random init, compared directly:
    uniform:  every unit decays at the same rate (mirrors the real
              architecture's coarse gate)
    spread:   units are assigned decay rates spread across [0, 1] (mirrors
              what `diag(1-alpha_t)` would allow if it were used)

Run
---
    python experiments/titans_per_unit.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DIM, HIDDEN = 64, 256
BREAK_THRESHOLD = 0.15
WRITE_LR = 0.01  # constant write strength, same in both conditions (0.05+ diverges to NaN -- no momentum/clipping here to stabilize a larger step, unlike the real library)


def init_weights(seed: int):
    g = torch.Generator().manual_seed(seed)
    w0 = torch.empty(DIM, HIDDEN)
    w1 = torch.empty(HIDDEN, DIM)
    torch.nn.init.xavier_uniform_(w0, generator=g)
    torch.nn.init.xavier_uniform_(w1, generator=g)
    return w0, w1


def mlp(x, w0, w1):
    return F.gelu(x @ w0) @ w1


def store_tracked_pairs(pairs: torch.Tensor, decay_per_unit: torch.Tensor, seed: int):
    """pairs: (n, dim). Each pair is its own key AND value. decay_per_unit:
    (hidden,) -- may be a constant-valued or spread-valued vector. Returns
    the final (w0, w1) after storing every pair in sequence."""
    w0, w1 = init_weights(seed)
    for t in range(pairs.shape[0]):
        x = pairs[t]
        w0_ = w0.detach().requires_grad_(True)
        w1_ = w1.detach().requires_grad_(True)
        pred = mlp(x, w0_, w1_)
        loss = (pred - x).pow(2).sum()
        grad_w0, grad_w1 = torch.autograd.grad(loss, (w0_, w1_))

        surprise_w0 = -WRITE_LR * grad_w0
        surprise_w1 = -WRITE_LR * grad_w1

        with torch.no_grad():
            w0 = (1 - decay_per_unit[None, :]) * w0 + surprise_w0
            w1 = (1 - decay_per_unit[:, None]) * w1 + surprise_w1

    return w0, w1


@torch.no_grad()
def recall_cosines(w0, w1, pairs):
    retrieved = mlp(pairs, w0, w1)
    r = retrieved / (retrieved.norm(dim=-1, keepdim=True) + 1e-9)
    v = pairs / (pairs.norm(dim=-1, keepdim=True) + 1e-9)
    return (r * v).sum(-1).numpy()


def ablate(w0, w1, unit):
    w0, w1 = w0.clone(), w1.clone()
    w0[:, unit] = 0.0
    w1[unit, :] = 0.0
    return w0, w1


def run_ablation_sweep(w0, w1, pairs):
    baseline = recall_cosines(w0, w1, pairs)
    drop = np.zeros((HIDDEN, pairs.shape[0]))
    for u in range(HIDDEN):
        aw0, aw1 = ablate(w0, w1, u)
        drop[u] = baseline - recall_cosines(aw0, aw1, pairs)
    return baseline, drop


def report(label: str, decay_per_unit: torch.Tensor, baseline, drop):
    print("=" * 70)
    print(f"{label}   decay range [{decay_per_unit.min():.2f}, {decay_per_unit.max():.2f}]"
          f"   {HIDDEN} units x {drop.shape[1]} pairs")
    print("=" * 70)
    print("baseline recall (cos) per pair, in store order:")
    print(" ", np.round(baseline, 3))

    hit = np.abs(drop) > BREAK_THRESHOLD
    hit_counts = hit.sum(axis=1)
    print(f"\nlargest single-unit effect on any pair: {np.abs(drop).max():.3f}  (threshold {BREAK_THRESHOLD})")
    print(f"units breaking 0 / 1 / >1 pairs hard:  "
          f"{(hit_counts == 0).sum()} / {(hit_counts == 1).sum()} / {(hit_counts > 1).sum()}")

    total_effect = np.abs(drop).sum(axis=1)
    print("\nmost causally important units:")
    for u in np.argsort(-total_effect)[:8]:
        hp = np.where(np.abs(drop[u]) > BREAK_THRESHOLD)[0]
        print(f"  unit {u:>4}  decay={decay_per_unit[u]:.2f}  total|drop|={total_effect[u]:.3f}"
              f"  breaks pairs {list(hp)}  row={np.round(drop[u], 2)}")


def main():
    n_pairs = 12
    seed = 0
    g = torch.Generator().manual_seed(seed)
    pairs = torch.randn(n_pairs, DIM, generator=g)

    uniform_decay = torch.full((HIDDEN,), 0.3)
    spread_decay = torch.linspace(0.02, 0.98, HIDDEN)

    for label, decay in [("UNIFORM decay (mirrors the real architecture)", uniform_decay),
                          ("SPREAD decay (mirrors what diag(1-alpha_t) allows)", spread_decay)]:
        w0, w1 = store_tracked_pairs(pairs, decay, seed)
        baseline, drop = run_ablation_sweep(w0, w1, pairs)
        report(label, decay, baseline, drop)
        print()


if __name__ == "__main__":
    main()
