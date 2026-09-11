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
| `titans_ablation.py` | causal follow-up to the proxy-dependent part above. `python experiments/titans_ablation.py [--train]`. Stores tracked key/value pairs and ablates one hidden unit at a time to see which pair's recall breaks — real intervention, not a correlational guess. |
| `../notebooks/marv_titans_ablation_colab.ipynb` | Colab version of the ablation test, with a replay-fidelity sanity check, baseline-recall plot, and a per-(unit, pair) drop heatmap. |

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

1. **Answer the storage question with ablation, not a proxy. — ANSWERED, 2026-09-11.**
   `titans_ablation.py`: store N *tracked* key→value pairs, ablate hidden units
   one at a time, measure which pair's recall breaks. The 2026-09-08 blocker
   (naive `functional_call` on one final weight state only matched true
   retrieval at cos ≈ 0.6) turned out to be a fidelity bug, not a structural
   one: calling the library's own public `NeuralMemory.retrieve_memories()`
   (which wraps `functional_call` with a pre-norm, multi-head split, q-norm,
   multihead RMSNorm, retrieve gate, and head merge that the naive version
   skipped) reproduces the model's true output at cos ≈ 1.0.
   With a faithful ablation in hand: **no single hidden unit localizes any
   one tracked pair**, trained or not — the largest single-unit effect on any
   pair's recall was ≈0.13 (cosine scale), and no unit's effect concentrates
   on one pair. Storage is genuinely **distributed / holographic**, causally
   confirmed, not the earlier proxy's guess. Trained-memory recall is also
   strongly recency-biased (last-stored pairs recall far better than early
   ones) — an independent, direct-recall confirmation of the forgetting curve
   from `titans_memdiff.py`.
   **New open question:** does a *group* ablation (top-k units most
   implicated in one pair, removed together) break that pair, even though no
   single one of them does? That would distinguish "small coalition" from
   "truly uniform."
2. **Scale.** `dim 512`, 2–4 memory layers, 500–2000 token documents. Does the
   forgetting curve stay exponential? Does distributed storage hold, or does
   a localization regime appear at a different scale? Does a survival-vs-
   distance curve and a capacity knee appear?
3. **Real vocabulary.** Wire the memory into a small LM (titans-pytorch MAC on
   char/byte enwik8 fits a T4) so a `describe_feature`-style logit lens reads
   what a unit promotes: "unit 33 now writes `Colchester`." Also lets the
   ablation test use real facts instead of random vectors.
4. **Package it.** If the analysis stabilises: a `marv` adapter for a plain-MLP
   memory + `marv.diff`-compatible snapshots, and a Colab notebook.
5. **The paper shape.** "Instrumenting test-time memory with feature-level
   diffs" — workshop-scale if the numbers show clean structure. The
   forgetting curve and the causal distributed-storage result both do now;
   a capacity knee (item 2) and a real-vocabulary readout (item 3) would
   carry it further.
