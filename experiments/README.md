# experiments/ — MARV × Titans (branch: `marv-titan`)

Prototype work applying MARV's weight-space primitives to a **test-time memory**
(Titans / TTT-style: a small MLP whose weights are updated by gradient descent
*during inference*). Nothing here is wired into the `marv` package yet.

## The idea

`marv.diff` compares two *training* checkpoints feature by feature
(`gate_cos` / `down_cos` / `gate_norm_ratio`). A Titans neural memory is an MLP
that changes while the model reads — and `titans-pytorch` exposes the
accumulated weight-delta at every chunk boundary in `state.updates`, i.e. a
stack of snapshots of the same evolving MLP. So MARV's diff applies directly:
**snapshot the memory early in a document, snapshot it at the end, diff per
hidden unit.** Then ask how much of an early write survives (the forget gate,
measured per unit) and — the open part — what each changed unit actually stored.

## What's here

| file | what |
|---|---|
| `titans_memdiff.py` | standalone prototype. `python experiments/titans_memdiff.py [--train]`. Needs `pip install titans-pytorch`. Runs on CPU or a Colab T4 (training is seconds either way). |
| `../notebooks/marv_titans_memdiff_colab.ipynb` | Colab version — explains how Titans works, trains a memory on recall, then diffs untrained vs trained with plots (collision histogram, write-concentration curve, forgetting curve). Imports the helpers from this file. |

## Prototype findings (2026-09-08, `dim 64 → 256 → 64`, 96-token random doc)

### Solid — metric-independent, stable across 3+ document seeds

| | untrained memory | trained on recall (MSE ≈ 0.03) |
|---|---|---|
| `norm_ratio` (end / early write) | **1.6 – 2.0** — writes accumulate | **0.05 – 0.06** — writes decay hard |
| first-chunk write, norm by chunk | 0.64 → 0.84 (holds / grows) | 0.85 → 0.03 (clean exponential, step ratio ≈ 0.5) |
| Gini of a single chunk's write over the 256 units | 0.12 | 0.04 |

1. **An untrained memory does not forget** — the forget gate is effectively off,
   writes pile on top of each other.
2. **A trained memory forgets exponentially** — first-chunk write down to ~4% of
   peak after six chunks, half-life ≈ 1 chunk. Tight curve, stable across seeds.
   (Decay *rate* is task-dependent — `--train` uses a crude short-sequence
   recall task and may have taught an unusually strong gate.)
3. **The write is diffuse in both** — Gini 0.04 – 0.12 ≈ near-uniform across all
   256 units. No sparse "this chunk → these few units" allocation, trained or not.
4. So the trained memory's apparent lack of write collisions is **forgetting,
   not clean storage**: by end-of-document it holds almost nothing (184 – 226 of
   256 units carry no identifiable content), so there is nothing left to collide.

### Proxy-dependent — treat as rough

The "which chunk's content did this unit store" analysis aligns a unit's
weight-delta against the *chunk-mean value vector* — the mean of 16 random unit
vectors, which is near-degenerate (points almost nowhere). So the specific
collision counts (`~193` untrained, `~1` trained) are shaky, and the trained
memory's peak-write content-alignment came out at ≈ 0 (indistinguishable from
the bad target). **Whether a trained memory localises storage at the moment of
writing is unanswered** — see roadmap 1.

The literature's "Titans memorises facts but free-form retrieval is only 0–40%"
(arXiv:2510.09551) is consistent with the forgetting result, but proving the
mechanism needs the ablation setup below.

## Literature position

5 web searches + the 2 closest papers, 2026-09-08. Could **not** find
feature-level weight-diff of a test-time memory across a document. Closest:

- **Titans: Learning to Memorize at Test Time** (arXiv:2501.00663) — downstream
  metrics only, no memory internals.
- **Titans Revisited** (arXiv:2510.09551, Oct 2025) — reproducibility +
  downstream ablations; explicitly does *not* inspect memory weights/neurons or
  diff them across a sequence.
- **Disentangling MLP Neuron Weights in Vocabulary Space** (arXiv:2604.06005) —
  logit-lens on MLP neurons, but *static* transformers.

Titans is ~20 months old. Treat "novel" as the **mechanism-level finding** — a
clean per-unit measurement of the weight-decay gate, with a trained/untrained
phase difference — not the tool. Caveat: search was not exhaustive; a workshop
paper could exist.

## Roadmap

1. **Answer the storage question with ablation, not a proxy.** Store N *tracked*
   key→value pairs; check recall per pair (`M(k_i)` vs `v_i`) as N grows;
   ablate hidden units one at a time and measure which units' removal kills a
   given pair's recall (`marv.suppress` / `rank_by_ablation_effect` on the live
   memory). Overlap between pairs' unit-sets is the *real* collision measurement.
   Blocker found 2026-09-08: `titans-pytorch` retrieves per chunk with per-chunk
   causal weights + head splitting; a naive `functional_call` on one final
   weight state only reproduces the true retrieval at cos ≈ 0.6, so the ablation
   has to go through the library's own retrieve path (inject modified weights
   into the state) or a faithful re-implementation.
2. **Scale.** `dim 512`, 2–4 memory layers, 500–2000 token documents. Does the
   forgetting curve stay exponential? Does a survival-vs-distance curve and a
   capacity knee appear?
3. **Real vocabulary.** Wire the memory into a small LM (titans-pytorch MAC on
   char/byte enwik8 fits a T4) so a `describe_feature`-style logit lens reads
   what a unit promotes: "unit 33 now writes `Colchester`." Also fixes the
   degenerate content target from finding-set 2.
4. **Package it.** If the analysis stabilises: a `marv` adapter for a plain-MLP
   memory + `marv.diff`-compatible snapshots, and a Colab notebook.
5. **The paper shape.** "Instrumenting test-time memory with feature-level
   diffs" — workshop-scale if the numbers show clean structure (the forgetting
   curve already does; a capacity knee and a real localisation measurement
   would carry it).
