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
| `titans_ablation.py` | causal follow-up to the proxy-dependent part above. `python experiments/titans_ablation.py [--train / --seed-stability]`. Stores tracked key/value pairs and ablates one hidden unit at a time to see which pair's recall breaks — real intervention, not a correlational guess. Also reports per-pair fragility (z-scored, collapsed across units) and a unit-importance concentration curve, and can run a multi-seed stability check on either. |
| `../notebooks/marv_titans_ablation_colab.ipynb` | Colab version of the ablation test, with a replay-fidelity sanity check, baseline-recall plot, a per-(unit, pair) drop heatmap, and the collapsed per-pair/per-unit views. |
| `titans_per_unit.py` | `python experiments/titans_per_unit.py`. Answers *why* no unit localizes anything in the real architecture, by testing what happens if the forget gate is allowed to vary per hidden unit instead of being one shared number. No training needed — decay rates are assigned directly, isolating that one variable. |

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
   on one pair. Trained-memory recall is strongly recency-biased (last-stored
   pairs recall far better than early ones) — an independent, direct-recall
   confirmation of the forgetting curve from `titans_memdiff.py`.
   A follow-up heatmap suggested a "fragile middle pair" (store-position 7–8)
   — a `--seed-stability` check across 5 seeds showed the top-fragile
   *position* moves every time (8, 7, 3, 8, 9 across seeds); **not a real
   effect, just noise from one run.** Correctly caught before it became a
   claim.

1b. **Why no localization — sourced from the real implementation and the
   paper, 2026-09-11.** Reading `neural_memory.py::store_memories` in full
   (not just `retrieve_memories`) shows the forget gate (`to_decay_factor`)
   and write strength (`to_adaptive_step`) are each a **single scalar per
   head per chunk**, broadcast identically onto all 256 hidden units — the
   architecture has no mechanism to forget or write one unit differently
   from another. Checked directly against the paper (arXiv:2501.00663, not
   just code comments): main-text eq. 13 writes `M_t = (1-α_t)M_{t-1} + S_t`
   with no per-unit index; Appendix C's more general derivation (eq. 32)
   writes it as `diag(1-α_t) M_t`, which *would* permit per-dimension decay —
   the released implementation collapses to the coarser scalar case. No
   sentence found stating why (a cheap controller + the parallel associative
   scan in §3.2 both plausibly favor it, but that's inference, not a quote).
   **Conclusion: "storage doesn't localize" is closer to a structural
   consequence of this specific gate's coarseness than an emergent discovery
   about interference.**
   Also found while checking this: `NeuralMemory`'s default memory model is
   `ResidualNorm(MemoryMLP(...))` — i.e. `output = LN(MLP(query))*gamma +
   query`, a residual skip straight from query to output. Zeroing the
   *entire* MLP for a trained memory left most pairs' recall almost
   unchanged (`titans_ablation.py`'s `check_residual` test) — meaning for
   most (non-fresh) pairs, "recall" was mostly the residual floor (~0.5
   cosine), not memorized content. Only the 1–2 most-recently-written pairs
   showed a real MLP contribution (0.96 → 0.59 when the MLP was zeroed).
   The forgetting curve is real, but it's better described as "a real,
   decaying MLP bonus on top of a constant content-independent floor,"
   which sharpens rather than undermines the original finding.

1c. **Does per-unit decay change the answer? — YES, confirmed across 8 seeds,
   2026-09-11.** `titans_per_unit.py` tests the `diag(1-α_t)` case directly:
   same 2-layer GELU memory MLP, same autoassociative store, but each of the
   256 units gets its **own fixed** decay rate instead of one shared value
   (an earlier attempt to *learn* per-unit decay end-to-end did not converge
   in a reasonable number of steps and was abandoned in favor of this more
   direct test — noted in the script's docstring). Result, uniform vs.
   spread decay, same pairs, same seed:

   | | uniform decay (mirrors real arch.) | spread decay (mirrors `diag(1-α_t)`) |
   |---|---|---|
   | max single-unit effect on any pair, across 8 seeds | 0.015 – 0.024 (never crosses 0.15) | 0.099 – 0.207 (crosses in 4/8 seeds) |
   | seeds with a unit breaking exactly 1 pair | 0 / 8 | 4 / 8 |

   A slow-decaying unit reliably ends up as the de facto home for one
   specific fact once units are allowed to differ; a uniform gate never
   produces this in any of 8 seeds. **This is the positive result: the
   earlier "no localization" finding is real for the actual released
   architecture, but is a consequence of its coarse gate, not an inherent
   property of test-time memory — give it per-unit forgetting and real
   localization appears.**

1d. **The editing demo — find it, edit it, measure the damage, 2026-09-11.**
   `titans_per_unit.py::edit_unit_to_target` does MARV's actual core move on
   a live memory instead of a frozen one: solve for a new output row on the
   unit localization says owns a fact, so that fact recalls an arbitrary new
   target, using only that unit's own weights. The edit is always exact
   (cos to the new target = 1.000, solved for directly) — the real question
   is collateral. Two units tested, picked by ablation-specificity (how
   little deleting them disturbs other pairs):

   | | ablation-specificity | largest collateral after editing |
   |---|---|---|
   | unit 4 / pair 0 | 1.75 | 0.473 |
   | unit 14 / pair 4 | **2.95** (nearly 2×) | **0.540** (worse) |

   A more ablation-specific unit produced *more* collateral, not less.
   Ablation-specificity measures how small a unit's existing contribution to
   other pairs is; editing-collateral measures how much that unit's
   *activation* fires for other pairs' own keys, independent of how small
   its old content there was — a unit can look "safe" by one measure and be
   risky by the other. Confirms, causally and on a live memory, the same
   thing MARV's own docs say about frozen models: "one neuron is shared by
   many unrelated facts, that is also where collateral damage comes from."

2. **Scale.** `dim 512`, 2–4 memory layers, 500–2000 token documents. Does the
   forgetting curve stay exponential? Does the per-unit-decay localization
   result (1c) hold or sharpen at scale? Does a survival-vs-distance curve
   and a capacity knee appear?
3. **Real vocabulary — the forgetting curve REVERSES on real text, confirmed
   2026-09-11.** `titans_real_text.py` wires a small `MemoryAsContextTransformer`
   (dim=64 to match the memory's own `dim_head`, one memory layer, no
   flex-attn — deliberately scaled way down from the library's own
   `train_mac.py` recipe of dim 384 / depth 8 / 100k batches / wandb, which
   is a real multi-hour+ run) and trains it on real enwik8 text, then reuses
   `titans_memdiff.py`'s early-vs-end snapshot diff on the memory reading a
   real held-out passage instead of 96 random vectors. `data/enwik8.gz` came
   from a local clone of `lucidrains/titans-pytorch`, no separate download.

   Two independent runs, same direction, effect **strengthening** with more
   training (rules out "undertrained artifact"):

   | | random-vector, untrained | random-vector, trained (toy recall task) | real text, 1000 steps (CPU) | real text, 3000 steps (T4, val loss 1.88) |
   |---|---|---|---|---|
   | norm_ratio (end/early) | ~1.7 | **~0.07** | 1.81 | **2.20** |
   | direction retained | ~0.47 | ~0.66 | 0.87 | **0.95** |
   | magnitude retained (min) | — | always < 1 | 1.30 | **1.66 (never shrinks)** |

   A memory trained on an actual language-modeling objective does **not**
   develop the aggressive exponential forgetting seen on the toy
   autoassociative-recall task — it behaves like the *untrained* case
   (writes accumulate), and more real training pushes this further, not
   less. This confirms a caveat that was in this README from the first
   commit: *"decay rate is task-dependent — `train_recall` uses a crude
   short-sequence recall task and may have taught an unusually strong
   gate."* The earlier "trained memory forgets exponentially" finding is
   real for that specific toy task, but is **not** a general property of
   trained Titans memories — plausible reading: predicting the next byte of
   real text rewards retaining context (topic, recent characters), while
   the toy recall task had nothing to gain from keeping anything beyond the
   immediate query.
   **Still open:** a `describe_feature`-style logit lens reading what a unit
   promotes, and re-running `titans_per_unit.py`'s per-unit-decay
   localization test on a real-text-trained memory instead of random
   vectors.
4. **Package it.** If the analysis stabilises: a `marv` adapter for a plain-MLP
   memory + `marv.diff`-compatible snapshots, and a Colab notebook.
5. **The paper shape.** "Instrumenting test-time memory with feature-level
   diffs, and showing its lack of localization is a gate-granularity
   artifact, not an inherent property" — workshop-scale if item 3 (real
   vocabulary) and item 2 (scale) hold up the story from toy random vectors.
