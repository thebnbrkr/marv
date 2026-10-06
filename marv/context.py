"""Contextual probes: run a real prompt through the live model and read the
result off the vindex, instead of querying with a bare embedding row.

`probe.describe()` / `probe.describe_entity()` query with `vindex.embed[...]`
directly -- the raw, un-contextualized embedding. A layer's gate rows are
defined in that layer's own transformed basis (after N layers of attention +
FFN have reshaped the residual), which a bare embedding was never rotated
into, so KNN hits against it tend to be weak. Running the prompt through the
model and reading its *actual* hidden state at that layer puts the query in
the right basis and gives much stronger matches.

The query at layer L is what layer L's FFN actually receives: the residual
after that layer's attention, passed through its pre-FFN norm (read with a
hook on `adapter.ffn_in`). That is the vector the gate rows multiply. HF's
`output_hidden_states` gives something else, the residual AFTER the FFN has
written and without the norm (and the last entry already final-normed), so
it is not used here.

Cosine similarity with a gate row is still only a proxy for "fires": in a
gated MLP a feature's activation is act(gate . x) * (up . x), and a negative
up . x makes the feature push its tokens DOWN. `active_features` reads the
real, signed activation instead and reports both directions.

The tool-calling helpers that used to live here moved to marv/toolcall.py,
which is now an optional domain layer on top of this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .extract import VindexLite
from .probe import describe_feature, logit_lens, top_features


@torch.no_grad()
def _last_inputs(model, tokenizer, prompt, layers, module_of, device):
    from .trace import _adapter, _inputs

    ad = _adapter(model)
    out, handles = {}, []

    def make(L):
        def pre(_m, args):
            out[L] = args[0][0, -1].float().cpu().numpy()
        return pre

    try:
        for L in layers:
            handles.append(module_of(ad, model, L).register_forward_pre_hook(make(L)))
        model(**_inputs(model, tokenizer, prompt, device))
    finally:
        for h in handles:
            h.remove()
    return out


def hidden_states_at_layers(model, tokenizer, prompt, layers: list[int], device: str = "cpu"):
    """Last-token FFN input at each requested layer: the normalised residual
    the layer's gate rows actually multiply. {layer: (hidden,) array}."""
    return _last_inputs(model, tokenizer, prompt, layers, lambda ad, m, L: ad.ffn_in(m, L), device)


def feature_activations_at_layers(model, tokenizer, prompt, layers: list[int], device: str = "cpu"):
    """Last-token feature activations at each requested layer, exactly as the
    FFN output projection receives them (act(gate . x) * (up . x) for a gated
    MLP). Signed. {layer: (intermediate,) array}."""
    return _last_inputs(model, tokenizer, prompt, layers, lambda ad, m, L: ad.ffn_out(m, L), device)


@dataclass
class ActiveFeature:
    layer: int
    feature: int
    activation: float  # signed; minus the baseline's when a baseline was given
    pushes_up: list[int]  # token ids this feature currently promotes
    pushes_down: list[int]  # token ids it currently suppresses


def active_features(model, tokenizer, vindex: VindexLite, prompt, layers: list[int], k: int = 10,
                    k_tokens: int = 5, baseline_prompt=None, device: str = "cpu") -> list[ActiveFeature]:
    """The features that actually fire on `prompt`, by |activation|, with
    what each one pushes up and down GIVEN the sign it fired with. With
    `baseline_prompt`, ranks by the change in activation instead (what the
    probe word adds). Sorted by |activation| across all `layers`."""
    acts = feature_activations_at_layers(model, tokenizer, prompt, layers, device)
    if baseline_prompt is not None:
        base = feature_activations_at_layers(model, tokenizer, baseline_prompt, layers, device)
        acts = {L: a - base[L] for L, a in acts.items()}
    rows = []
    for L, a in acts.items():
        for f in np.argsort(-np.abs(a))[:k]:
            col = np.sign(a[f]) * vindex.down[L][:, f]
            up, _ = logit_lens(vindex, col, k=k_tokens)
            down, _ = logit_lens(vindex, -col, k=k_tokens)
            rows.append(ActiveFeature(L, int(f), float(a[f]), [int(t) for t in up], [int(t) for t in down]))
    rows.sort(key=lambda r: -abs(r.activation))
    return rows


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
