"""Contextual probes: run a real prompt through the live model and read the
result off the vindex, instead of querying with a bare embedding row.

`probe.describe()` queries with `vindex.embed[token_id]` directly -- the raw,
un-contextualized embedding. A layer's gate rows are defined in that layer's
own transformed basis (after N layers of attention + FFN have reshaped the
residual), which a bare embedding was never rotated into, so KNN hits against
it tend to be weak (low cosine similarity, mediocre labels). Running the
prompt through the model and reading its *actual* hidden state at that layer
puts the query in the right basis and gives much stronger matches.

The functions here happen to default to tool-call prompts (DEFAULT_TOOL_PROMPTS)
but nothing about them is tool-call-specific -- swap in prompts for any other
behavior (safety, math, factual recall, ...) or any two same-architecture
checkpoints, tool calling not required.
"""
from __future__ import annotations

import numpy as np
import torch

from .extract import VindexLite
from .probe import describe_feature, logit_lens, top_features

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


def describe_prompt(
    vindex: VindexLite,
    model,
    tokenizer,
    prompt: str,
    layers: list[int],
    k_features: int = 10,
    k_tokens: int = 5,
    device: str = "cpu",
    baseline_prompt: str | None = None,
):
    """Contextual version of `probe.describe()`: the query at each layer is
    the model's own hidden state after actually processing `prompt`, not a
    raw embedding row -- the fix for the "France gets 0.11 cosine similarity"
    problem.

    Pass `baseline_prompt` (the same prompt with the probe word/topic removed,
    e.g. prompt="I want to talk about France", baseline_prompt="I want to
    talk about") to subtract that hidden state first. Without it, a single
    dominant, roughly prompt-invariant direction -- a "massive activation" /
    outlier channel, a documented phenomenon in small transformers -- can
    swamp the query regardless of content, since the per-word signal is a
    small perturbation on top of a much larger shared-template direction.
    Differencing against the templated baseline cancels that shared part out
    and isolates what's actually specific to the probe word.

    Returns the same shape as `probe.describe()`:
    {layer: [(feature_idx, cos_sim, top_token_ids, top_logits), ...]}.
    """
    hidden = hidden_states_at_layers(model, tokenizer, prompt, layers, device=device)
    baseline = (
        hidden_states_at_layers(model, tokenizer, baseline_prompt, layers, device=device)
        if baseline_prompt is not None
        else None
    )
    out = {}
    for layer, vector in hidden.items():
        query = vector - baseline[layer] if baseline is not None else vector
        hits = []
        for feature_idx, sim in top_features(vindex, layer, query, k=k_features):
            tok_ids, logits = describe_feature(vindex, layer, feature_idx, k=k_tokens)
            hits.append((feature_idx, sim, tok_ids, logits))
        out[layer] = hits
    return out
