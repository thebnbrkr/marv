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

Editing demo (2026-09-11): the actual MARV move -- find a fact, edit it,
measure the damage -- applied to a live memory instead of a frozen one.
`edit_unit_to_target()` solves for a new output row on one unit so a
tracked pair recalls an arbitrary new target instead of its original value,
using only that unit's own weights (found via the localization test above).
The edit itself is always mathematically exact (cos to the new target =
1.000, since it's solved for directly) -- the real question is collateral:
does retargeting the "owning" unit disturb the OTHER stored pairs?

Result: yes, substantially, and picking a MORE ablation-specific unit does
NOT reduce it -- it got worse. Two candidates, same setup:

    unit 4  / pair 0  (ablation-specificity 1.75): largest collateral 0.473
    unit 14 / pair 4  (ablation-specificity 2.95): largest collateral 0.540

Ablation-specificity measures how small a unit's EXISTING contribution to
other pairs is (so deleting it barely moves them). Editing-collateral
measures something different: how much that unit's ACTIVATION fires for
OTHER pairs' own keys, regardless of how small its old content there was --
if it fires for many keys, injecting a big new output row disturbs all of
them, whatever its previous weights used to encode. These are different
properties of a unit, and one does not predict the other -- a stronger,
more specific version of "one neuron carries many unrelated facts" than
either the ablation or the editing result alone would have shown.

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


def edit_unit_to_target(w0: torch.Tensor, w1: torch.Tensor, unit: int, key: torch.Tensor,
                         target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The actual MARV move, not just deletion: solve for a NEW output row
    on `unit` so that retrieving with `key` produces `target` instead of
    whatever it currently retrieves -- using only that one unit's own
    weights, the same neuron the localization test showed owns this fact.
    Leaves `unit`'s gate column (w0) untouched -- editing changes what a
    unit says when it fires, not whether it fires."""
    w0e, w1e = w0.clone(), w1.clone()
    ablated_w0, ablated_w1 = ablate(w0, w1, unit)
    with torch.no_grad():
        r_ablated = mlp(key, ablated_w0, ablated_w1)  # what's retrieved WITHOUT this unit's help
        activation = F.gelu(key @ w0[:, unit])
        if abs(activation.item()) < 1e-6:
            raise ValueError(f"unit {unit} barely activates for this key "
                              f"(activation={activation.item():.2e}) -- pick a unit that actually fires for it")
        new_row = (target - r_ablated) / activation
    w1e[unit, :] = new_row
    return w0e, w1e


def run_editing_demo(pairs: torch.Tensor, decay_per_unit: torch.Tensor, seed: int,
                      unit: int, edit_pair_idx: int, target: torch.Tensor):
    """Find it, edit it, measure the damage -- MARV's core loop, on a LIVE
    memory instead of a frozen one. Store the pairs normally, pick a unit
    the localization test says owns `edit_pair_idx`, retarget ONLY that
    unit so the edited pair now recalls `target`, then check the fact
    changed and everything else didn't."""
    w0, w1 = store_tracked_pairs(pairs, decay_per_unit, seed)
    baseline = recall_cosines(w0, w1, pairs)

    key = pairs[edit_pair_idx]
    ew0, ew1 = edit_unit_to_target(w0, w1, unit, key, target)
    edited = recall_cosines(ew0, ew1, pairs)
    edited_to_target = F.cosine_similarity(mlp(key, ew0, ew1), target, dim=0).item()

    print("=" * 70)
    print(f"EDITING DEMO -- unit {unit}, retargeting pair {edit_pair_idx}")
    print("=" * 70)
    print("recall (cos) per pair, before vs. after the edit:")
    print("  pair   before   after   moved?")
    for i in range(len(pairs)):
        moved = "  <-- EDITED" if i == edit_pair_idx else ("  <-- collateral" if abs(edited[i] - baseline[i]) > 0.1 else "")
        print(f"  {i:>4}   {baseline[i]:>6.3f}  {edited[i]:>6.3f}{moved}")
    print(f"\nedited pair's recall of its NEW target: {edited_to_target:.3f} (cos to the fact we edited in, want high)")

    others = [i for i in range(len(pairs)) if i != edit_pair_idx]
    max_collateral = np.abs(np.array(edited)[others] - np.array(baseline)[others]).max()
    print(f"largest change on any OTHER pair (collateral damage): {max_collateral:.3f}")


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

    print("\n" + "#" * 70)
    print("# EDITING DEMO -- find it, edit it, measure the damage")
    print("#" * 70)
    # unit 4 was flagged above as cleanly owning pair 0 (ablation-specificity 1.75)
    run_editing_demo(pairs, spread_decay, seed=seed, unit=4, edit_pair_idx=0, target=pairs[5])
    print()
    # unit 14/pair 4 on a different seed: nearly 2x more ablation-specific (2.95) --
    # a stricter selection criterion, tested to see if it reduces collateral (it doesn't)
    g7 = torch.Generator().manual_seed(7)
    pairs7 = torch.randn(n_pairs, DIM, generator=g7)
    run_editing_demo(pairs7, spread_decay, seed=7, unit=14, edit_pair_idx=4, target=pairs7[8])


if __name__ == "__main__":
    main()
