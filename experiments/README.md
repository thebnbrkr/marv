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
hidden unit.** Then ask what each changed unit stored, whether chunks collide on
the same unit, and how much of an early write survives (the forget gate,
measured per unit).

## What's here

| file | what |
|---|---|
| `titans_memdiff.py` | standalone prototype. `python experiments/titans_memdiff.py [--train]`. Needs `pip install titans-pytorch`. Runs on CPU or a Colab T4 (training is seconds either way). |
| `../notebooks/marv_titans_memdiff_colab.ipynb` | Colab version — explains how Titans works, trains a memory on recall, then diffs untrained vs trained with plots (collision histogram, write-concentration curve, forgetting curve). Imports the helpers from this file. |

## Prototype findings (2026-09-08, `dim 64 → 256 → 64`, 96-token random doc)

| | untrained memory | trained on recall (MSE ≈ 0.03) |
|---|---|---|
| hidden units that move | 256 / 256 | 256 / 256 |
| `gate_cos` median | +0.34 | +0.60 |
| `norm_ratio` (end / early) | ~1.7 (writes grow) | ~0.10 (writes decay hard) |
| write collisions (units carrying >1 chunk's content) | **199 / 256** | **4 / 256** |
| early-written units overwritten by the end | — | **20 / 20** |
| one unit's down-norm trajectory | rise / decay / rise | 0.92 → 0.51 → 0.26 → 0.17 → 0.12 → 0.09 → 0.08 |

Read: an **untrained** memory is total superposition — every chunk smears across
all units, ~78% of units carry multiple chunks. **Training** buys content
separation (collisions collapse) but also teaches an aggressive forget gate —
old writes decay ~10× over six chunks. The literature's "Titans memorises facts
but free-form retrieval is only 0–40%" (arXiv:2510.09551) looks driven by
*aggressive weight-decay of old writes*, not superposition alone.

**Big caveats:** toy scale (one memory, dim 64), no real vocabulary, `--train`
uses a crude autoassociative-recall task, and with `norm_ratio ≈ 0.1` the
trained-memory diff should be re-run measuring each unit's write at its *peak*
chunk rather than at the end.

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

Titans is ~20 months old. Treat "novel" as the **mechanism-level finding**
(dense collision when untrained; decay-driven forgetting when trained), not the
tool. Caveat: search was not exhaustive; a workshop paper could exist.

## Roadmap

1. **Trained vs untrained, properly.** Re-run the collision analysis measuring
   each unit's write at its peak chunk. Does training localise storage, or just
   forget faster?
2. **Scale.** `dim 512`, 2–4 memory layers, 500–2000 token documents. Does the
   collision rate hold? Does a forgetting *curve* (survival vs distance) emerge?
3. **Real vocabulary.** Wire the memory into a small LM (titans-pytorch MAC on
   char/byte enwik8 fits a T4) so a `describe_feature`-style logit lens reads
   what a unit promotes: "unit 33 now writes `Colchester`."
4. **Package it.** If the analysis stabilises: a `marv` adapter for a plain-MLP
   memory + `marv.diff`-compatible snapshots, and a Colab notebook.
5. **The paper shape.** "Instrumenting test-time memory with feature-level
   diffs" — workshop-scale if the trained-memory numbers show clean structure
   (a forgetting curve, a capacity knee, a localisation-vs-superposition
   phase change with scale).
