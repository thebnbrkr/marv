"""Contextual probes: run a real prompt through the live model and read the
result off the vindex, instead of querying with a bare embedding row.

`probe.describe()` / `probe.describe_entity()` query with `vindex.embed[...]`
directly -- the raw, un-contextualized embedding. A layer's gate rows are
defined in that layer's own transformed basis (after N layers of attention +
FFN have reshaped the residual), which a bare embedding was never rotated
into, so KNN hits against it tend to be weak. Running the prompt through the
model and reading its *actual* hidden state at that layer puts the query in
the right basis and gives much stronger matches.

Nothing here is architecture-specific: any model whose HF forward accepts
`output_hidden_states=True` works (Llama, Mistral, Qwen, TinyLlama, ...).
The tool-calling helpers that used to live here moved to marv/toolcall.py,
which is now an optional domain layer on top of this module.
"""
from __future__ import annotations

import numpy as np
import torch

from .extract import VindexLite
from .probe import describe_feature, top_features


def hidden_states_at_layers(model, tokenizer, prompt: str, layers: list[int], device: str = "cpu"):
    """Last-token residual stream at each requested layer -- a cheap
    stand-in for a full forward-hook capture system."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the embedding output; layer i's output is hidden_states[i+1]
    return {L: out.hidden_states[L + 1][0, -1, :].float().cpu().numpy() for L in layers}


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

    Pass `baseline_prompt` (the same prompt with the probe word/topic
    removed, e.g. prompt="I want to talk about France",
    baseline_prompt="I want to talk about") to subtract that hidden state
    first. Without it, a single dominant, roughly prompt-invariant direction
    -- a "massive activation" / outlier channel, documented in small
    transformers -- can swamp the query regardless of content. Differencing
    against the templated baseline cancels that shared part out.

    Returns {layer: [(feature_idx, cos_sim, top_token_ids, top_logits), ...]}.
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
