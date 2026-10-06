# MARV — Model Architecture Research via Vindex

MARV turns a transformer's gated FFN weights into an inspectable index
(a **vindex**) so you can browse what a small model knows, diff two
checkpoints at the level of individual FFN features, edit the live model
without fine-tuning, and **measure what the edit broke** — on a laptop or
a single Colab T4:

- **Extract** a gated FFN's `gate_proj`/`down_proj` weights into a plain
  numpy structure (the vindex). Fits-in-RAM path *and* a streaming path
  that reads one layer at a time straight from `.safetensors` for models
  bigger than RAM.
- **Probe** it: which FFN features fire for a given input, and what tokens
  each feature promotes (gate-KNN + logit-lens — the `describe` path).
  `describe_entity("France")` gives you the LARQL-style browse view with
  no knowledge-graph pipeline.
- **Diff two checkpoints** (e.g. base vs. fine-tuned) at the level of
  individual FFN features — the closest thing here to "git diff for what a
  model learned." Same machinery diffs fp16 vs int4.
- **Edit + measure**: hide or ablate a feature on the *live* model, then
  run a tagged probe battery before/after and get a table of exactly what
  flipped, what degraded, and what held. This is the point of MARV — on a
  small model you can afford the dense evaluation.

The vindex name/idea come from [LARQL](https://github.com/chrishayuk/larql)
(a much larger Rust system). MARV shares none of its code: one compressed
`.npz`, held in RAM, Llama-style FFN only. See `AGENTS.md` for the full map.

## Suppression is not deletion

`vindex.suppressed` hides a feature from `describe_*` — a **retrieval-layer
filter**. The weights are untouched; a real forward pass still fires the
feature. To change behaviour you intervene on the live model:
`marv.suppress` (forward hooks, reversible) or `marv.ablate` (zeros
`down_proj[:, f]`, permanent). Because one neuron is shared by many
unrelated facts, that is also where collateral damage comes from —
`marv.study_edit` measures it.

## Why this works on Llama-style models without an architecture registry

SmolLM2 / Llama / Mistral / Qwen / TinyLlama are plain gated-FFN dense
models: `model.model.layers[i].mlp.{gate,up,down}_proj` + SiLU. One adapter
(`marv/arch.py::LlamaStyleFFN`) covers all of them — `hidden_size` and
`intermediate_size` are read from `config.json`, not hardcoded. A
genuinely different FFN (Gemma GeGLU, MoE) later means one more
`ArchAdapter` subclass, not rearchitecting the pipeline.

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```python
import marv
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(name)
tok = AutoTokenizer.from_pretrained(name)

vindex = marv.extract(model, model_name=name)
marv.build_down_meta(vindex)                 # once: makes describe_feature a lookup

# "What does the model associate with France, and which features carry it?"
for row in marv.describe_entity(vindex, tok, "France"):
    print(row)                               # L24 f4123 sim=0.41 -> ['Paris', 'France', ...]
```

### Edit and measure

```python
# the ranked (layer, feature) constellation carrying "France"
feats = [(r.layer, r.feature) for r in marv.constellation(vindex, tok, "France")[:4]]

battery = [
    marv.Probe("The capital of France is", "Paris",  ("target",)),
    marv.Probe("The capital of Italy is",  "Rome",   ("neighbour",)),
    marv.Probe("The Eiffel Tower is in",   "Paris",  ("neighbour",)),
    # ... plus many unrelated probes tagged ("control",)
]

rep = marv.study_edit(model, tok, marv.suppress(model, feats), battery)
rep.show()            # only what moved, + "N unchanged"
rep.show(full=True)   # every probe
```

### Diff two checkpoints

```python
vb, vt = marv.extract(base_model), marv.extract(tuned_model)
for d in marv.most_changed(marv.diff(vb, vt), k=10):
    print(f"L{d.layer} f{d.feature_idx}: gate_cos={d.gate_cos_sim:.3f} "
          f"down_cos={d.down_cos_sim:.3f} norm_ratio={d.gate_norm_ratio:.2f}")
```

### Check before you trust: diagnostics and tracing

Ported from [marv-hyena](https://github.com/thebnbrkr/marv-hyena), the MARV fork that studies Evo 2 (a DNA model).
Every one of these fixed a real mistake there, and none is specific to DNA:

```python
texts = ["The sun rises in the east.", "Water boils at 100 degrees."]      # ordinary text

# 1. Does the model still work under the edit? (an edit that breaks the model looks very "effective")
rep = marv.study_edit(model, tok, marv.suppress(model, feats), battery, health_texts=texts)
rep.show()                        # adds: health 0.61 -> 0.58, or "MODEL BROKEN, results uninterpretable"

# 2. Which single layers break the model on their own? Keep those on in group ablations.
[r for r in marv.load_bearing(model, tok, texts) if r.broken]

# 3. Does one layer dominate the residual stream? Then direct attribution will only name that layer.
marv.find_bottlenecks(model, tok, "The capital of France is")

# 4. Which layers pushed the answer up, directly? The rows add up EXACTLY to the real logit difference.
d = marv.decompose_logit(model, tok, "The capital of France is", " Paris", " Rome")
d.show(); d.check_error           # ~0, otherwise don't trust the rows

# 5. Total effects (including indirect paths): swap each layer's attention / MLP output between prompts
m = marv.logit_diff_metric(tok, " Rome", " Paris")
marv.patch_sweep(model, tok, "The capital of Italy is", "The capital of France is", m).show()

# 6. At which layer does "Italy" hand its information to the answer position?
pos = tok("The capital of Italy is")["input_ids"].index(tok(" Italy", add_special_tokens=False)["input_ids"][0])
marv.trace_by_depth(model, tok, "The capital of Italy is", "The capital of France is", position=pos, metric=m)

# 7. Neurons that never fire (their "top examples" are noise), and a shuffled-weights baseline
marv.dead_features(model, tok, texts)
null = marv.null_model(model)     # run the same analysis on it; whatever also shows up there is an artifact
```

**Direct vs. total.** `decompose_logit` measures what each layer's output pushes *by itself*. `patch_sweep`,
`trace_by_depth` and `mean_ablate` run real forward passes and include every indirect effect. They answer
different questions. On Evo 2 the direct answer was "block 30: 100%", which was true and useless, because block
30's output was ~10⁵× larger than every other layer's.

See `scripts/demo_smollm2.py` and the notebooks:

| notebook | what |
|---|---|
| `notebooks/marv_llama32_1b_colab.ipynb` | end-to-end on Llama-3.2-1B: extract, browse, contextual constellation, edit, base-vs-instruct weight diff |
| `notebooks/marv_qwen25_05b_colab.ipynb` | Qwen2.5-0.5B: causal constellation edit, per-layer breakdown of what carries the fact, Pareto frontier, then two weight diffs off one base (post-training, code training) |
| `notebooks/marv_edit_eval_colab.ipynb` | evaluating an edit: causal constellation, efficacy vs. specificity, the Pareto frontier, `suppress` vs `ablate` vs `steer` |
| `notebooks/marv_history_colab.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thebnbrkr/marv/blob/main/notebooks/marv_history_colab.ipynb) | History: teach SmolLM2-135M a made-up fact, record every checkpoint, then `log`, `bisect`, `gate` and `blame` the run (Colab GPU, ~10 min) |
| `notebooks/marv_tests_colab.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thebnbrkr/marv/blob/main/notebooks/marv_tests_colab.ipynb) | the test suite plus exactness, target-token, lineage, provenance and scale checks on real SmolLM2-135M (CPU, ~3 min) |
| `notebooks/marv_amd135m_edit_colab.ipynb` | knowledge-editing AMD-Llama-135M: narrow vs wide control battery (why 8 controls isn't enough), per-layer breakdown, base-vs-code weight diff |
| `notebooks/marv_smollm2_colab.ipynb` | original SmolLM2 base vs tool-tuned probe |

## Layout

```
marv/
  arch.py       architecture adapters: every module path MARV uses; register_adapter for new ones
  extract.py    vindex extraction (in-RAM + streaming), save/load, layer bands
  probe.py      static weight-space analysis: gate-KNN, logit-lens, describe*
  context.py    contextual probing (real forward pass, through attention)
  diff.py       per-feature weight-space delta between two checkpoints
  edit.py       live-model interventions: suppress / ablate / steer / constellation
  evaluate.py   probe batteries: run_battery / diff_battery / study_edit (health_texts=)
  trace.py      direct attribution (decompose_logit, self-checked), patch_sweep, trace_by_depth, mean_ablate
  diagnostics.py health, load_bearing, write_norms / find_bottlenecks, dead_features, null_model
  toolcall.py   optional tool-calling prompt scaffolds over context.py
  heatmap.py, layer_heatmap.py, clustering.py   polysemanticity + activation heatmaps
scripts/demo_smollm2.py       end-to-end: 135M base vs 135M function-calling
notebooks/                    T4-ready
```

## Models

Any Llama-style gated-FFN checkpoint: SmolLM2, TinyLlama, Llama 2/3,
Mistral, Qwen 2/2.5. `describe_prompt` / `study_edit` need the model in
RAM; `extract_streaming` + weight-space probing do not.

## Tests

```bash
python -m pytest -q     # 42 tests on a synthetic tiny-Llama, no network
```
