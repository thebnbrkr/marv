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

See `scripts/demo_smollm2.py` and the notebooks:

| notebook | what |
|---|---|
| `notebooks/marv_qwen3_colab.ipynb` | end-to-end on Qwen3-0.6B: extract, browse, edit, ablate, Base-vs-post-trained diff |
| `notebooks/marv_edit_eval_colab.ipynb` | evaluating an edit: efficacy vs. specificity, the Pareto frontier, `suppress` vs `ablate` vs `steer`, per-layer localization |
| `notebooks/marv_smollm2_colab.ipynb` | original SmolLM2 base vs tool-tuned probe |

## Layout

```
marv/
  arch.py       architecture adapter (module-name -> weight tensors)
  extract.py    vindex extraction (in-RAM + streaming), save/load, layer bands
  probe.py      static weight-space analysis: gate-KNN, logit-lens, describe*
  context.py    contextual probing (real forward pass, through attention)
  diff.py       per-feature weight-space delta between two checkpoints
  edit.py       live-model interventions: suppress / ablate / steer / constellation
  evaluate.py   probe batteries: run_battery / diff_battery / study_edit
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
python -m pytest -q     # synthetic tiny-Llama, no network
```
