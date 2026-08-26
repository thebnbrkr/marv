# MARV — Model Architecture Research via Vindex

MARV turns a transformer's gated FFN weights into an inspectable index
(a **vindex**) so you can browse what a small model knows and diff two
checkpoints at the level of individual FFN features, on a single Colab T4:

- Extract a gated FFN's `gate_proj`/`down_proj` weights into a plain
  numpy structure (the vindex).
- Probe it: which FFN features fire for a given input, and what tokens
  each feature promotes (gate-KNN + logit-lens — the `describe` path).
- **Diff two checkpoints** (e.g. base vs. fine-tuned) at the level of
  individual FFN features — the closest thing here to "version control
  for what a model knows."
- Use both of the above around a model's tool-calling behavior
  specifically: which features fire at the tool-call decision point,
  and does fine-tuning move them.

MARV stays deliberately minimal: no custom binary/mmap format, no
query language, no serving layer, no quantization. A 135M/1.7B HF
checkpoint fits fully in RAM, so a loaded `transformers` model plus
numpy is enough — none of that engineering is needed at this scale.

## Why this works on SmolLM2 without an architecture registry

[SmolLM2](https://huggingface.co/HuggingFaceTB) is a plain Llama-style
dense model: `model.model.layers[i].mlp.{gate,up,down}_proj` + SiLU,
the same module names as Llama/Mistral/Qwen2. One adapter
(`marv/arch.py::LlamaStyleFFN`) covers all of them — `hidden_size` and
`intermediate_size` are read from `config.json`, not hardcoded. Adding
a genuinely different FFN shape (MoE, GeGLU) later means adding one
more `ArchAdapter` subclass, not rearchitecting the pipeline.

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from marv.extract import extract
from marv.probe import describe_feature, top_features

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(name)
tok = AutoTokenizer.from_pretrained(name)

vindex = extract(model)

# "What does feature 42 at layer 6 promote?"
tokens, logits = describe_feature(vindex, layer=6, feature_idx=42)
print(tok.batch_decode(tokens[:, None]))
```

See `scripts/demo_smollm2.py` for the full base-vs-tool-tuned comparison,
and `notebooks/marv_smollm2_colab.ipynb` to run it on a T4.

## Layout

```
marv/
  arch.py       architecture adapter (module-name → weight tensors)
  extract.py    vindex extraction + save/load
  probe.py      gate-KNN + logit-lens ("describe")
  diff.py       per-feature delta between two checkpoints ("diff")
  toolcall.py   tool-call-specific probes
scripts/
  demo_smollm2.py   end-to-end: 135M-Instruct vs 135M-Function-Calling vs 1.7B-Instruct
notebooks/
  marv_smollm2_colab.ipynb   same demo, T4-ready
```

## Models used

| Model | Role |
|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | small base — has *not* been tuned for tool calling |
| `gvij/SmolLM2-135M-Function-Calling` | same size, community tool-call fine-tune — the "after" checkpoint |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | larger model with *official* tool-calling support (trained on Argilla Synth-APIGen-v0.1) |

All three fit comfortably in fp16 on a T4 (16 GB) with room for
activations; no quantization needed.
