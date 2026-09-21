# AGENTS.md

Guidance for AI coding agents (and humans) working in this repo.

## What MARV is

MARV turns a transformer's **gated FFN weights** into a small, inspectable
numpy structure (a **vindex**) and gives you tools to browse it, diff two
checkpoints at the level of individual FFN features, edit the live model
without fine-tuning, and **measure what an edit broke**.

It is a microscope for **small** models (135M–~3B) run on a laptop or a
free Colab T4. The design bet: at this scale you can afford exhaustive
experiments — enumerate every feature, run hundreds of probes per edit,
chart the efficacy/specificity frontier — that interpretability work on
7B+ models can only sample.

The `vindex` name and concept are borrowed from
[LARQL](https://github.com/chrishayuk/larql) (a much larger Rust system —
"the model is a database"). MARV shares none of LARQL's code or file
format: MARV's vindex is one compressed `.npz`, held in RAM, Llama-style
FFN only. Credit LARQL for the idea; MARV takes it somewhere narrower and
more measurement-focused.

## Core object: `VindexLite` (`marv/extract.py`)

A dataclass, one per checkpoint:

| field | shape | meaning |
|---|---|---|
| `gate[layer]` | `(intermediate, hidden)` | `gate_proj` weight — **one row per FFN feature** (a "neuron") |
| `down[layer]` | `(hidden, intermediate)` | `down_proj` weight — **one column per feature** |
| `embed` / `lm_head` | `(vocab, hidden)` | token lookup / unembedding |
| `final_norm_weight`, `norm_eps` | — | the real final-RMSNorm gain, for a faithful logit lens |
| `layer_bands` | dict | heuristic `syntax` / `knowledge` / `output` depth split |
| `suppressed` | `set[(layer, feature)]` | features hidden from `top_features` / `describe_*` — a **retrieval filter, not a weight edit** |
| `down_meta_tokens` / `down_meta_scores` | per layer `(intermediate, k)` | optional precomputed logit-lens cache |

Persisted with `save(path)` / `VindexLite.load(path)` — a single `.npz`.
Old `.npz` files missing the newer fields still load (defaults fill in).

### Build a vindex

- `extract(model)` — walk a loaded HF model. Needs the model in RAM;
  keeps a live-model path open for contextual probing.
- `extract_streaming(model_dir)` — read one layer at a time straight from
  `.safetensors` shards, never instantiate the torch model. For
  checkpoints bigger than RAM. No live model afterwards, so
  `marv.context` contextual probing is unavailable on the result.

## Module layout

```
marv/
  arch.py        ArchAdapter / LlamaStyleFFN — module-name -> (gate, up, down, embed, lm_head, norm)
  extract.py     VindexLite, extract(), extract_streaming(), default_layer_bands()
  probe.py       STATIC weight-space analysis (no forward pass, no attention):
                   top_features   — gate-KNN, respects vindex.suppressed
                   logit_lens     — project a residual direction through (norm +) lm_head
                   build_down_meta(device="cuda") — precompute promoted tokens (GPU path
                                    is ~50x faster on a big vocab)
                   describe_feature / describe / describe_entity
  context.py     CONTEXTUAL probing (real forward pass, through attention):
                   hidden_states_at_layers, describe_prompt (baseline-differenced)
  diff.py        weight-space checkpoint diff: FeatureDelta(gate_cos_sim, down_cos_sim,
                 gate_norm_ratio), most_changed(), per_layer_score()
  edit.py        interventions on a LIVE model:
                   suppress(model, feats)  — forward hooks, reversible (context manager)
                   ablate(model, feats)    — zero down_proj[:, f] in place, permanent
                   restore(model, saved)   — undo ablate
                   steer(model, L, v, a)   — add a*v to the residual after layer L
                   constellation(vindex, tok, entity, model=, prompt=) — ranked (layer,
                                    feature) set carrying a fact; pass model= for a
                                    contextual (sharp) query instead of the bare embedding
  evaluate.py    Probe (target_ids: candidate first-token ids, " Paris" vs "Paris"),
                 run_battery, diff_battery, study_edit, suppression_frontier,
                 suppression_by_layer (per-layer breakdown of a constellation edit —
                   which layers of the constellation are load-bearing vs just KNN hits;
                   cumulative= for the depth the effect saturates at),
                 frontier_table (a suppression_frontier sweep as plain (n, drop, drop, moved) rows),
                 rank_by_ablation_effect (causal constellation — rank candidates by
                   measured target-prob drop when suppressed alone),
                 BatteryDiff.show(full=) / .metrics() (per-tag efficacy vs collateral)
  trace.py       RESIDUAL-STREAM TRACING (activation-space). Llama-style layers add exactly two
                 writes each: attn (self_attn output) and mlp (mlp output); final = embed + all writes.
                   capture_writes  — every write at chosen positions; .reconstruction_error() ~0
                   decompose_logit — DIRECT attribution of a logit difference through the frozen
                                     final RMSNorm; rows sum exactly to the real value (.check_error)
                   replace_outputs / mean_writes / mean_ablate — component-level interventions
                   patch_sweep     — TOTAL effect: swap each component source->target (same length)
                   trace_by_depth  — patch the whole residual at one position after each layer:
                                     where information leaves that position
  diagnostics.py CHECKS TO RUN BEFORE TRUSTING A RESULT (from marv-hyena / Evo 2):
                   health (next-token acc + logprob on ordinary text; study_edit(health_texts=)),
                   load_bearing (single components that break the model alone),
                   write_norms / find_bottlenecks (one write dominating the residual),
                   dead_features (neurons that never fire: their top examples are noise),
                   null_model (weights shuffled per tensor: a baseline for artifacts)
  batteries.py   curated probe sets: WORLD_CAPITALS + SCIENCE/LEXICAL/MATH/HISTORY/
                 COMMONSENSE lists, broad_controls() (~110 tagged by sub-domain),
                 capital_edit_battery(country, capital, neighbours). Pure data.
  toolcall.py    OPTIONAL domain layer — tool-calling prompt scaffolds over context.py.
                 Nothing requires this file; ignore it for non-tool-call work.
  heatmap.py / layer_heatmap.py / clustering.py   polysemanticity + activation heatmaps
scripts/demo_smollm2.py       end-to-end base vs tool-tuned
notebooks/*.ipynb             T4-ready
tests/test_arch.py            adapter + extract + diff + heatmap (synthetic Llama)
tests/test_edit.py            edit + evaluate + describe (synthetic Llama)
tests/test_diagnostics.py     trace + diagnostics, each checked against an exact answer
```

## The two kinds of analysis — keep them straight

1. **Weight-space** (`probe.py`, `diff.py`): read the stored matrices,
   project columns through `lm_head`, cosine-compare gate rows. **No model
   execution, no attention.** `build_down_meta` is pure `lm_head @ down[layer]`.
2. **Activation-space** (`context.py`, `edit.py`, `evaluate.py`): run a
   real prompt through the live HF model. Goes through attention and every
   layer. Needs `transformers` + the model in RAM.

`describe_entity` is weight-space (queries the bare embedding — fast,
weaker matches). `describe_prompt` is activation-space (queries the real
hidden state — slower, stronger matches, needs the model).

## Invariants — do not break these

- **A "feature" is one raw MLP neuron at one layer.** Not an SAE
  dictionary latent. Feature 134 at L3 and feature 134 at L18 are
  unrelated. Features are polysemantic — one fires for many unrelated
  things. `heatmap.polysemantic_features` exists to measure this.
- **`vindex.suppressed` never touches weights.** It filters
  `top_features` / `describe_*` only. A real forward pass of the model
  still fires the feature. To affect a forward pass use `edit.suppress`
  (hooks) or `edit.ablate` (weight write). This distinction is the point,
  not a bug — document it wherever it comes up.
- **Facts are constellations, not neurons.** "France -> Paris" is a
  weighted pattern across several features in the knowledge band, and each
  of those features also serves other facts. That overlap is where edit
  collateral damage comes from. Default to operating over a **band**
  (`vindex.band("knowledge")`), not a single layer.
- **`extract()` copies tensors with `.copy()`** so a vindex can never
  alias live model weights. Keep this.
- **Every exact decomposition carries its own check.** `decompose_logit` reports
  `actual` (read from the real logits) and `check_error`; `Writes` reports
  `reconstruction_error()`. Never report rows whose check failed, and never remove a
  check to make something pass. A failing check usually means the architecture isn't
  really Llama-style (e.g. extra post-norms).
- **Direct ≠ total.** `decompose_logit` is direct only; `patch_sweep` /
  `trace_by_depth` / ablations are total. Label which one a number is.
- **Run a health check with every ablation or edit.** Stubbing something load-bearing
  breaks the whole model, and a broken model fails every test at once. That looks
  like a finding and isn't. Use `study_edit(..., health_texts=)` or `health()`.
- **Check for bottlenecks before trusting direct attribution** (`find_bottlenecks`).
  If one write dominates the residual, direct attribution names only that layer.
- **Prefer mean-ablation to zeroing** for whole components (`mean_ablate`). Zeroing
  pushes the residual off-distribution.
- **Hooks are always removed**, even on exceptions. Every intervention is a context
  manager or a try/finally.
- **Llama-style FFN only, for now.** `mlp.{gate,up,down}_proj` + SiLU.
  Adding Gemma / GPT-2 / MoE means a new `ArchAdapter` subclass and
  storing `activation` / `embed_scale` / `logit_softcap` on the vindex —
  not special-casing inside `probe.py` / `edit.py`.

## Typical workflow

```python
import marv
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model, tok = AutoModelForCausalLM.from_pretrained(name), AutoTokenizer.from_pretrained(name)

v = marv.extract(model, model_name=name)
marv.build_down_meta(v)                       # once — makes describe_feature a lookup

# what does the model associate with an entity, and which features carry it
for row in marv.describe_entity(v, tok, "France"):
    print(row)                                # L24 f4123 sim=0.41 -> ['Paris', 'France', ...]

# pick a constellation, measure what suppressing it does
feats = [(r.layer, r.feature) for r in marv.constellation(v, tok, "France")[:4]]
battery = [
    marv.Probe("The capital of France is", "Paris", ("target", "capital")),
    marv.Probe("The capital of Italy is",  "Rome",  ("neighbour", "capital")),
    marv.Probe("The capital of Spain is",  "Madrid",("neighbour", "capital")),
    marv.Probe("The Eiffel Tower is in",   "Paris", ("neighbour", "geography")),
    # ... dozens of unrelated probes tagged "control"
]
rep = marv.study_edit(model, tok, marv.suppress(model, feats), battery)
rep.show()             # only what moved + "N unchanged"
rep.show(full=True)    # every probe
```

## Diff two checkpoints (the git-diff-after-finetuning use case)

```python
vb = marv.extract(base_model);  vt = marv.extract(tuned_model)
for d in marv.most_changed(marv.diff(vb, vt), k=10):
    print(f"L{d.layer} f{d.feature_idx}: gate_cos={d.gate_cos_sim:.3f} down_cos={d.down_cos_sim:.3f}")
```

This is **weight-space** and per-feature — LARQL's `DIFF` is fact-space
(it compares resolved edges / precomputed top tokens), so MARV's diff is
genuinely a different tool, the right one for "what did fine-tuning move".
Same machinery works for `diff(vindex_f16, vindex_int4)` — which features
a quantiser damaged.

## Build / test

```bash
pip install -r requirements.txt
python -m pytest -q                 # tests/ (42 tests), synthetic, no network
python scripts/demo_smollm2.py      # downloads SmolLM2 checkpoints
```

Tests build a tiny synthetic `LlamaForCausalLM` — no checkpoint download,
runs anywhere `transformers` + `torch` are installed. New behaviour that
can be checked on synthetic weights should get a test there.

## Sibling project

[marv-hyena](https://github.com/thebnbrkr/marv-hyena) applies MARV's approach to Evo 2 (StripedHyena:
Hyena convolutions + attention, DNA). `trace.py` and `diagnostics.py` are ported from it. The
Hyena-specific parts (filter reach, exact lag-band splits of convolutions, first-layer enumeration)
stay there.

## Not in scope (say so if asked)

- SAE / dictionary learning (raw neurons only, by design)
- running inference from the vindex (MARV relies on `transformers` for
  forward passes; there is no WalkFfn)
- a query language, a server, on-disk mmap format, MoE
- true "unlearning" — MARV does factual editing / suppression, not
  removing a distribution of knowledge
