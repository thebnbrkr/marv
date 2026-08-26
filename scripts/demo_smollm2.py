"""End-to-end MARV demo: base vs. tool-tuned SmolLM2, plus a size comparison.

Run directly (CPU/MPS/local GPU), or paste into
notebooks/marv_smollm2_colab.ipynb cells to run on a T4.

    python scripts/demo_smollm2.py
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marv.diff import diff, most_changed
from marv.extract import extract
from marv.toolcall import (
    DEFAULT_TOOL_PROMPTS,
    compare_tool_prompt,
    describe_prompt,
    hidden_states_at_layers,
)

BASE_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"
TUNED_135M = "gvij/SmolLM2-135M-Function-Calling"
INSTRUCT_1_7B = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Concept probes -- what fires for these words, and what does it promote.
PROBE_WORDS = ["weather", "function", "France"]
PROBE_TEMPLATE = "I want to talk about {word}"
PROBE_BASELINE = "I want to talk about"


def load(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    model.eval()
    return model, tok


def concept_probe(vindex, model, tok, label: str, words=PROBE_WORDS, device: str = "cpu"):
    """Run each word through the live model in a short template sentence
    (contextual hidden state, not a bare embedding row -- see
    marv.toolcall.describe_prompt), differenced against the template alone to
    cancel out the dominant prompt-invariant direction, KNN over the gate
    rows of a late layer (logit lens is only reliably legible in roughly the
    last third of the model), and label the firing features by what they
    promote."""
    print(f"\n--- concept probe: {label} ---")
    late_layers = [round(vindex.num_layers * frac) for frac in (0.7, 0.85)]
    late_layers = sorted(set(l for l in late_layers if l < vindex.num_layers))
    for word in words:
        prompt = PROBE_TEMPLATE.format(word=word)
        hits = describe_prompt(
            vindex, model, tok, prompt, layers=late_layers, k_features=3, k_tokens=3,
            device=device, baseline_prompt=PROBE_BASELINE,
        )
        print(f"'{word}':")
        for layer, features in hits.items():
            for feature_idx, sim, tok_ids, logits in features:
                words_out = tok.batch_decode([[t] for t in tok_ids])
                print(f"  L{layer} f{feature_idx} (sim={sim:.2f}) -> {words_out}")


def main():
    print("Loading base (no tool-call tuning):", BASE_135M)
    base_model, base_tok = load(BASE_135M)
    vindex_base = extract(base_model, model_name=BASE_135M)

    print("Loading tool-tuned checkpoint:", TUNED_135M)
    tuned_model, tuned_tok = load(TUNED_135M)
    vindex_tuned = extract(tuned_model, model_name=TUNED_135M)

    concept_probe(vindex_base, base_model, base_tok, "base")
    concept_probe(vindex_tuned, tuned_model, tuned_tok, "tool-tuned")

    print("\n--- diff: which features moved most during tool-call fine-tuning ---")
    deltas = diff(vindex_base, vindex_tuned)
    for d in most_changed(deltas, k=10):
        print(
            f"L{d.layer} f{d.feature_idx}: gate_cos={d.gate_cos_sim:.3f} "
            f"down_cos={d.down_cos_sim:.3f} norm_ratio={d.gate_norm_ratio:.2f}"
        )

    print("\n--- tool-call decision point: base vs. tuned logit lens ---")
    # Last third of the model only -- logit lens on early/mid-layer residuals
    # generally doesn't decode to legible tokens (the stream hasn't rotated
    # into an output-ready basis yet), so those layers mostly show noise.
    start = max(1, round(vindex_base.num_layers * 0.66))
    probe_layers = list(range(start, vindex_base.num_layers, 3))
    for prompt in DEFAULT_TOOL_PROMPTS:
        print(f"\nprompt: {prompt!r}")
        h_base = hidden_states_at_layers(base_model, base_tok, prompt, probe_layers)
        h_tuned = hidden_states_at_layers(tuned_model, tuned_tok, prompt, probe_layers)
        comparison = compare_tool_prompt(vindex_base, vindex_tuned, h_base, h_tuned)
        for layer, result in comparison.items():
            base_words = base_tok.batch_decode([[t] for t, _ in result["base_top"][:3]])
            tuned_words = tuned_tok.batch_decode([[t] for t, _ in result["tuned_top"][:3]])
            print(f"  L{layer}: base={base_words}  tuned={tuned_words}")

    print("\nDone. To bring in the 1.7B official tool-calling model for a size")
    print(f"comparison, load {INSTRUCT_1_7B!r} the same way and re-run the")
    print("logit-lens probe (note: diff() needs matching layer counts, so it")
    print("can't be diffed against the 135M models directly -- probe only).")


if __name__ == "__main__":
    main()
