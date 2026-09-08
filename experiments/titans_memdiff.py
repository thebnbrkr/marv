"""MARV's feature-level weight-diff, applied to a Titans neural memory as it reads.

STATUS: prototype / WIP (branch: marv-titan). Not wired into the `marv` package yet.

Idea
----
MARV's core diff primitive (`marv.diff` -> per-feature gate_cos / down_cos /
gate_norm_ratio) normally compares two *training* checkpoints. A Titans /
test-time-training memory is a small MLP whose weights are updated by gradient
descent *during inference*. titans-pytorch exposes the accumulated weight-delta
at every chunk boundary in `state.updates`, i.e. a stack of snapshots of the
same evolving MLP -- exactly what a diff wants two of.

So: snapshot the fast-weight MLP early in a document, snapshot it at the end,
diff per hidden unit. Then ask which chunk's content each changed unit absorbed,
whether chunks collide on the same unit, and how much of an early write survives
to the end (the forget gate, measured per unit).

Prototype findings (2026-09-08, dim 64 -> 256 -> 64, 96-token random document).
See experiments/README.md for the full write-up + the sanity-check results.

SOLID (metric-independent, stable across seeds):
* untrained memory does NOT forget -- writes accumulate, norm_ratio end/early
  ~1.6-2.0.
* trained memory (--train, recall MSE ~0.03) forgets EXPONENTIALLY -- first-chunk
  write down to ~4% of peak after 6 chunks, step ratio ~0.5, half-life ~1 chunk.
  Decay rate is task-dependent (crude recall task may teach a strong gate).
* the write is DIFFUSE in both -- Gini of one chunk's write over the 256 units
  is 0.04-0.12 (near-uniform). No sparse per-chunk unit allocation.
* so the trained memory's apparent lack of write collisions is FORGETTING, not
  clean storage: by end-of-document it holds almost nothing.

PROXY-DEPENDENT (rough -- the "which chunk did unit u store" step aligns a unit's
weight-delta against the chunk-MEAN value = mean of 16 random vectors, which is
near-degenerate). The collision counts printed by section 2 are indicative only;
whether a trained memory localises storage at write time is UNANSWERED -- needs
the ablation setup in README roadmap 1.

Literature position (5 web searches + 2 papers, 2026-09-08)
----------------------------------------------------------
Could not find feature-level weight-diff of a test-time memory across a
document in the literature. Closest: "Titans Revisited" (arXiv:2510.09551,
Oct 2025) -- reproducibility + downstream ablations, explicitly does NOT
inspect memory weights/neurons or diff them across a sequence. Titans is ~20
months old. Treat "novel" as: the *mechanism-level finding* (dense collision,
forgetting curve), not the tool.

Run
---
    pip install titans-pytorch
    python experiments/titans_memdiff.py            # untrained memory
    python experiments/titans_memdiff.py --train    # train on recall first, then diff
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from titans_pytorch import NeuralMemory

DIM, HIDDEN, CHUNK = 64, 256, 16
DOC_LEN = 96


def _cos(a, b, axis):
    return np.sum(a * b, axis) / (
        np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + 1e-9
    )


def train_recall(mem: NeuralMemory, steps: int = 400, bs: int = 16, n: int = 32,
                 lr: float = 1e-3, device: str = "cpu") -> None:
    """Train the memory's slow/outer weights on autoassociative recall:
    the memory queried with token i should return token i."""
    opt = torch.optim.Adam(mem.parameters(), lr=lr)
    for step in range(steps):
        x = torch.randn(bs, n, DIM, device=device)
        retrieved, _ = mem(x)
        loss = torch.nn.functional.mse_loss(retrieved, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"  train step {step:4d}  recall MSE {loss.item():.4f}")


def snapshots(mem: NeuralMemory, seq: torch.Tensor):
    """Return the per-chunk stacks of accumulated weight-deltas:
    U0 (chunks, DIM, HIDDEN)  -- input/'gate' columns
    U1 (chunks, HIDDEN, DIM)  -- output/'down' rows
    plus the values the memory assigned to each token."""
    _, state = mem(seq)
    U0 = state.updates["model.weights.0"].detach()[0].cpu().numpy()
    U1 = state.updates["model.weights.1"].detach()[0].cpu().numpy()
    with torch.no_grad():
        vals = mem.to_values(seq)[0].detach().cpu().numpy()
    vals /= np.linalg.norm(vals, axis=1, keepdims=True) + 1e-9
    return U0, U1, vals


def section_diff(U0, U1):
    """marv.diff.FeatureDelta, per hidden unit: early-in-doc vs end-of-doc."""
    g_in, g_out = U0[1], U0[-1]
    d_in, d_out = U1[1], U1[-1]
    gate_cos = _cos(g_in, g_out, axis=0)
    down_cos = _cos(d_in, d_out, axis=1)
    nr = (np.linalg.norm(g_out, axis=0) + 1e-9) / (np.linalg.norm(g_in, axis=0) + 1e-9)
    moved = gate_cos < 0.99

    print("=" * 70)
    print("1. DIFF THE MEMORY  --  early in the document  vs  end")
    print("=" * 70)
    print(f"hidden units moved (gate_cos < 0.99): {moved.sum()} / {len(gate_cos)}")
    print(f"gate_cos  min {gate_cos.min():+.3f}   median {np.median(gate_cos):+.3f}")
    print(f"down_cos  min {down_cos.min():+.3f}   median {np.median(down_cos):+.3f}")
    print(f"norm_ratio on moved units  mean {nr[moved].mean():.2f}   max {nr.max():.2f}")
    print("\nmost-changed units (marv.most_changed):")
    print(f"  {'unit':>4} {'gate_cos':>9} {'down_cos':>9} {'norm_ratio':>11}")
    for u in np.argsort(gate_cos)[:8]:
        print(f"  {u:>4} {gate_cos[u]:>+9.3f} {down_cos[u]:>+9.3f} {nr[u]:>11.2f}")
    return moved


def section_collisions(U1, vals, moved):
    print("\n" + "=" * 70)
    print("2. WHICH CHUNK DID EACH CHANGED UNIT ABSORB?   (write collisions)")
    print("=" * 70)
    incr = np.diff(U1, axis=0)
    incr_norm = np.linalg.norm(incr, axis=2)
    nvc = min(incr.shape[0], math.ceil(DOC_LEN / CHUNK))
    incr, incr_norm = incr[:nvc], incr_norm[:nvc]
    cval = vals[: nvc * CHUNK].reshape(nvc, CHUNK, DIM).mean(1)
    cval /= np.linalg.norm(cval, axis=1, keepdims=True) + 1e-9
    idir = incr / (np.linalg.norm(incr, axis=2, keepdims=True) + 1e-9)
    align = np.einsum("khd,kd->kh", idir, cval)

    changed = np.where(moved)[0]
    assigned = collisions = 0
    for u in changed:
        strong = [k for k in range(nvc)
                  if incr_norm[k, u] > 0.5 * incr_norm[:, u].max() and align[k, u] > 0.15]
        if strong:
            assigned += 1
            collisions += len(strong) > 1
    print(f"changed units whose write aligns with a chunk's content: {assigned} / {len(changed)}")
    print(f"units written by >1 chunk with aligned content (collisions): {collisions}")
    for k in range(nvc):
        s = np.sort(incr_norm[k])[::-1]
        share = s[:16].sum() / (s.sum() + 1e-9)
        active = int((incr_norm[k] > 0.1 * incr_norm[k].max()).sum())
        print(f"  chunk {k}: top-16 units hold {share:5.1%} of the write   (active units: {active})")


def section_forgetting(U1):
    print("\n" + "=" * 70)
    print("3. FORGETTING  --  how much of an early write survives to the end")
    print("=" * 70)
    incr_norm = np.linalg.norm(np.diff(U1, axis=0), axis=2)
    early = np.argsort(incr_norm[0])[::-1][:20]
    w_early, w_end = U1[1, early, :], U1[-1, early, :]
    scos = _cos(w_early, w_end, axis=1)
    smag = np.linalg.norm(w_end, axis=1) / (np.linalg.norm(w_early, axis=1) + 1e-9)
    print(f"the 20 units the first chunk wrote hardest:")
    print(f"  direction retained (cos): mean {scos.mean():+.3f}  min {scos.min():+.3f}")
    print(f"  magnitude retained (ratio): mean {smag.mean():.2f}  min {smag.min():.2f}  max {smag.max():.2f}")
    print(f"  substantially overwritten (mag<0.6 or cos<0.5): {int(np.sum((smag < 0.6) | (scos < 0.5)))} / 20")
    u = early[0]
    traj = np.linalg.norm(U1[:, u, :], axis=1)
    print(f"\n  unit {u} down-vector norm by chunk: " + "  ".join(f"c{k}:{traj[k]:.2f}" for k in range(len(traj))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train on recall before diffing")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mem = NeuralMemory(dim=DIM, chunk_size=CHUNK).to(device)
    if args.train:
        print(f"training the memory on autoassociative recall ({args.steps} steps, {device})...")
        train_recall(mem, steps=args.steps, device=device)
        print()

    seq = torch.randn(1, DOC_LEN, DIM, device=device)
    U0, U1, vals = snapshots(mem, seq)
    print(f"document: {DOC_LEN} tokens, {U0.shape[0]} weight snapshots   "
          f"memory MLP: {DIM}->{HIDDEN}->{DIM}   trained: {args.train}\n")

    moved = section_diff(U0, U1)
    section_collisions(U1, vals, moved)
    section_forgetting(U1)


if __name__ == "__main__":
    main()
