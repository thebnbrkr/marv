"""Tool-call-specific probes: does fine-tuning move the FFN features around
the model's tool-call decision point, and does the logit-lens answer there
flip toward the right tokens?

This ties together extract.py (get the vindex), probe.py (logit lens) and
diff.py (which features moved) around prompts that should trigger tool use.
Swap DEFAULT_TOOL_PROMPTS for your own tool schema / few-shot format.
"""
from __future__ import annotations

import numpy as np
import torch

from .extract import VindexLite
from .probe import logit_lens, top_features

# Deliberately generic -- point these at whatever tool-call scaffold the
# model you're testing actually expects (e.g. SmolLM2's <tool_call> tags).
DEFAULT_TOOL_PROMPTS = [
    "What's the weather in Paris? Use the get_weather function.",
    "Convert 10 miles to kilometers using the unit_converter tool.",
]


def hidden_states_at_layers(model, tokenizer, prompt: str, layers: list[int], device: str = "cpu"):
    """Last-token residual stream at each requested layer -- a cheap
    stand-in for a full forward-hook capture system."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the embedding output; layer i's output is hidden_states[i+1]
    return {L: out.hidden_states[L + 1][0, -1, :].float().cpu().numpy() for L in layers}


def compare_tool_prompt(
    vindex_base: VindexLite,
    vindex_tuned: VindexLite,
    hidden_base: dict[int, np.ndarray],
    hidden_tuned: dict[int, np.ndarray],
    k: int = 5,
):
    """Logit-lens the same prompt's residual through both checkpoints' unembed
    at matching layers -- did fine-tuning change what the model 'wants to
    say' at the tool-call decision point, and in which layer does that first
    show up.
    """
    results = {}
    for layer in sorted(set(hidden_base) & set(hidden_tuned)):
        idx_b, logits_b = logit_lens(vindex_base, hidden_base[layer], k=k)
        idx_t, logits_t = logit_lens(vindex_tuned, hidden_tuned[layer], k=k)
        results[layer] = {
            "base_top": list(zip(idx_b.tolist(), logits_b.tolist())),
            "tuned_top": list(zip(idx_t.tolist(), logits_t.tolist())),
        }
    return results


def firing_features_at_layer(vindex: VindexLite, hidden: np.ndarray, layer: int, k: int = 10):
    """Which FFN features fire on this residual, at this layer -- feed this
    the same `hidden_states_at_layers` output used above, and cross-reference
    the feature indices against `diff.most_changed()` to see whether
    fine-tuning touched the features actually active at the decision point.
    """
    return top_features(vindex, layer, hidden, k=k)
