"""Causal ablation test for a Titans NeuralMemory: does a specific hidden
unit provably store a specific fact?

titans_memdiff.py's "which chunk did unit u store" analysis (section 2) is a
correlational proxy -- it aligns a unit's weight-delta against the chunk-mean
of random value vectors, which is near-degenerate (see experiments/README.md,
commit 33c2700). This script replaces correlation with causation: store N
distinct, individually-tracked key/value pairs, then zero one hidden unit at
a time and re-measure recall of every pair. A unit whose removal breaks ONE
pair's recall and leaves the rest untouched would be a real, causally-
verified localized store; a unit whose removal breaks MANY pairs at once
would be a real, causally-verified collision.

Fixes the retrieval-fidelity blocker from README roadmap item 1: the earlier
prototype called `functional_call(mem.memory_model, weights, q)` directly,
skipping the pre/post-processing `NeuralMemory.retrieve_memories()` does
around that call (retrieve_norm, multi-head split, q-norm, multihead
rmsnorm, the retrieve gate, head merge) -- that mismatch, not the per-chunk
weight structure, is what made it only match true retrieval at cos ~0.6.
This script calls the public `retrieve_memories()` method itself instead, so
every ablated-recall number is exactly what the model would actually
produce. Verified: replaying `state.updates` through `retrieve_memories`
reproduces the model's own output at cos ~1.0 (see `check_replay_fidelity`).

Result so far (2026-09-11, dim 64 -> 256 -> 64, 12 tracked pairs, trained on
the same autoassociative recall task as titans_memdiff.py):
* trained-memory recall is strongly recency-biased -- the last few pairs
  stored recall at cos 0.88-0.97, the earliest at cos 0.17-0.6. Independent
  confirmation of titans_memdiff.py's forgetting-curve finding via a
  completely different method (direct recall, not weight-snapshot norms).
* single-unit ablation never breaks any one pair: the largest recall drop
  from removing any single unit was ~0.13 (cosine scale -1..1), and no
  unit's effect concentrates on one pair -- every unit nudges several pairs
  a little. NO evidence of per-unit localized storage, causally verified
  (not a proxy). This says the earlier "diffuse write" finding is real: this
  memory's storage is distributed/holographic, not modular.
* untrained memory: same test, no drop clears 0.14 either -- expected, an
  untrained memory has no real recall function to localize in the first
  place (baseline recall ~0, no better than chance).

Open: does a *group* ablation (the top-k units most implicated in a pair,
removed together) break that pair, even though no single one of them does?
That would distinguish "distributed across a small coalition" from "truly
uniform across all 256 units."

Run
---
    pip install titans-pytorch
    python experiments/titans_ablation.py             # untrained memory
    python experiments/titans_ablation.py --train      # train on recall first
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from titans_pytorch import NeuralMemory

from titans_memdiff import train_recall

DIM, HIDDEN = 64, 256
GATE_KEY, DOWN_KEY = "model.weights.0", "model.weights.1"
BREAK_THRESHOLD = 0.15


def store_tracked_pairs(mem: NeuralMemory, n_pairs: int, dim: int, device: str, seed: int):
    """Store n_pairs distinct random vectors one at a time (chunk_size=1 ->
    one weight snapshot per pair). Each vector is its own key AND value,
    the same autoassociative convention train_recall() trains on. Returns
    the pairs and the single final weight snapshot after storing all of
    them (drops the per-chunk trajectory -- we only need "now")."""
    g = torch.Generator(device=device).manual_seed(seed)
    pairs = torch.randn(1, n_pairs, dim, device=device, generator=g)
    with torch.no_grad():
        _, state = mem(pairs)
    weights = {k: v[:, -1].clone() for k, v in state.updates.items()}
    return pairs, weights


def ablate(weights: dict, unit: int) -> dict:
    """Zero hidden unit `unit`'s input column (gate weight) and output row
    (down weight) -- removes it from both writing and reading."""
    w = {k: v.clone() for k, v in weights.items()}
    w[GATE_KEY][..., unit] = 0.0
    w[DOWN_KEY][..., unit, :] = 0.0
    return w


@torch.no_grad()
def recall_cosines(mem: NeuralMemory, weights: dict, pairs: torch.Tensor) -> np.ndarray:
    """Cosine sim between each pair's true value and what the memory
    actually retrieves for that pair's key, through the library's real
    retrieve_memories() path -- not a hand-rolled functional_call."""
    retrieved = mem.retrieve_memories(pairs, weights)
    r = retrieved[0].float().cpu().numpy()
    v = pairs[0].float().cpu().numpy()
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return (r * v).sum(axis=1)


@torch.no_grad()
def check_replay_fidelity(mem: NeuralMemory, seq: torch.Tensor) -> float:
    """Sanity check: does calling retrieve_memories() on the full per-chunk
    weight trajectory reproduce exactly what mem(seq) itself produced? If
    this isn't ~1.0, nothing else in this file should be trusted."""
    true_retrieved, state = mem(seq)
    replay = mem.retrieve_memories(seq, {k: v.clone() for k, v in state.updates.items()})
    cos = (true_retrieved[0] * replay[0]).sum(-1) / (
        true_retrieved[0].norm(dim=-1) * replay[0].norm(dim=-1) + 1e-9
    )
    return float(cos.min())


def run_ablation_sweep(mem, weights, pairs):
    baseline = recall_cosines(mem, weights, pairs)
    drop = np.zeros((HIDDEN, pairs.shape[1]))
    for u in range(HIDDEN):
        drop[u] = baseline - recall_cosines(mem, ablate(weights, u), pairs)
    return baseline, drop


def report(trained: bool, baseline: np.ndarray, drop: np.ndarray):
    print("=" * 70)
    print(f"trained={trained}   {HIDDEN} units x {drop.shape[1]} tracked pairs")
    print("=" * 70)
    print("baseline recall (cos) per pair, in store order (last = most recent):")
    print(" ", np.round(baseline, 3))

    hit = np.abs(drop) > BREAK_THRESHOLD
    hit_counts = hit.sum(axis=1)
    print(f"\nlargest single-unit effect on any pair: {np.abs(drop).max():.3f}  (threshold {BREAK_THRESHOLD})")
    print(f"units breaking 0 / 1 / >1 pairs hard:  "
          f"{(hit_counts == 0).sum()} / {(hit_counts == 1).sum()} / {(hit_counts > 1).sum()}")

    total_effect = np.abs(drop).sum(axis=1)
    print("\nmost causally important units (by total |drop| across all pairs):")
    for u in np.argsort(-total_effect)[:8]:
        print(f"  unit {u:>4}  total|drop|={total_effect[u]:.3f}  row={np.round(drop[u], 2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train on recall before testing")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mem = NeuralMemory(dim=DIM, chunk_size=1).to(device)
    if args.train:
        print(f"training the memory on autoassociative recall ({args.steps} steps, {device})...")
        train_recall(mem, steps=args.steps, device=device)
        print()

    fidelity = check_replay_fidelity(mem, torch.randn(1, 20, DIM, device=device))
    print(f"replay-fidelity check (should be ~1.0): {fidelity:.4f}\n")

    pairs, weights = store_tracked_pairs(mem, args.n_pairs, DIM, device, args.seed)
    baseline, drop = run_ablation_sweep(mem, weights, pairs)
    report(args.train, baseline, drop)


if __name__ == "__main__":
    main()
