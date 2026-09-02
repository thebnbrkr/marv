"""Interventions on a *live* HF model, for measuring what an edit does.

marv's vindex-level suppression (`VindexLite.suppressed`) is only a filter
on top_features / describe_* -- it does not touch the weights and does not
change a forward pass. To measure the *behavioural* effect of hiding or
removing a feature you have to intervene on the real model. These helpers
do that:

  suppress(model, feats)  -- torch forward hooks, reversible (context manager)
  ablate(model, feats)    -- zero down_proj[:, f] in place, permanent (returns
                             a handle for restore())
  steer(model, L, v, a)   -- add a*v to the residual after layer L

MARV's "feature" f at layer L is one column of that layer's gated MLP:
gated by gate_proj[f], scaled by up_proj[f], written through down_proj[:, f].
Zeroing the activation immediately before down_proj removes its entire
contribution to the residual stream for that pass -- and because the same
column is (almost always) shared by several unrelated facts, that is also
where the collateral damage comes from. `marv.evaluate.study_edit` is the
tool for measuring it.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def _mlp(model, layer: int):
    return model.model.layers[layer].mlp


@contextmanager
def suppress(model, features):
    """Zero the listed (layer, feature) MLP activations for the duration of
    the `with` block. Reversible -- hooks are removed on exit.

        with suppress(model, [(24, 4123), (25, 991)]):
            out = model.generate(**inputs)
    """
    by_layer: dict[int, list[int]] = {}
    for layer, feat in features:
        by_layer.setdefault(int(layer), []).append(int(feat))

    handles = []

    def make_hook(feats):
        idx = torch.tensor(feats, dtype=torch.long)

        def pre_hook(module, args):
            (x,) = args
            x = x.clone()
            x[..., idx.to(x.device)] = 0
            return (x,)

        return pre_hook

    try:
        for layer, feats in by_layer.items():
            h = _mlp(model, layer).down_proj.register_forward_pre_hook(make_hook(feats))
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


def ablate(model, features):
    """PERMANENT (in-memory): zero down_proj[:, f] so feature f contributes
    nothing on any path -- dense forward pass, generate, export. Returns a
    dict of the removed columns; pass it to restore() to undo.
    """
    saved: dict[tuple[int, int], torch.Tensor] = {}
    with torch.no_grad():
        for layer, feat in features:
            layer, feat = int(layer), int(feat)
            w = _mlp(model, layer).down_proj.weight  # (hidden, intermediate)
            saved[(layer, feat)] = w[:, feat].clone()
            w[:, feat] = 0
    return saved


def restore(model, saved):
    """Undo ablate(): put the saved down_proj columns back."""
    with torch.no_grad():
        for (layer, feat), col in saved.items():
            _mlp(model, layer).down_proj.weight[:, feat] = col.to(
                _mlp(model, layer).down_proj.weight.device
            )


@contextmanager
def steer(model, layer: int, direction, alpha: float = 1.0):
    """Add `alpha * direction` to the residual stream right after `layer`,
    for the duration of the block. `direction` is a length-hidden_size
    array/tensor.
    """
    vec = torch.as_tensor(direction, dtype=torch.float32)

    def hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * vec.to(h.device, h.dtype)
            return (h,) + tuple(output[1:])
        return output + alpha * vec.to(output.device, output.dtype)

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def constellation(
    vindex,
    tokenizer,
    entity,
    *,
    model=None,
    prompt: str | None = None,
    baseline_prompt: str | None = None,
    band: str = "knowledge",
    per_layer: int = 4,
    k_tokens: int = 5,
    embed_scale: float = 1.0,
    device: str = "cpu",
):
    """The ranked set of (layer, feature) that carry `entity`'s associations
    -- the thing you would suppress/ablate to change what the model says
    about it, and whose overlap with *other* entities is the collateral risk.

    Default: bare-embedding gate-KNN (fast, no forward pass). On many models
    this is noisy -- pass `model` (and optionally `prompt`, default a short
    template about `entity`, plus `baseline_prompt` to difference against) to
    query the model's *actual* hidden state instead. Much sharper.

    Returns Association rows sorted by similarity; slice `[:n]` for a minimal
    constellation. For the *causal* version -- rank by measured suppression
    effect on a target prompt -- see marv.evaluate.rank_by_ablation_effect.
    """
    from .probe import Association, describe_feature, describe_entity, top_features

    if model is None:
        return describe_entity(
            vindex, tokenizer, entity, band=band, k_features=per_layer,
            k_tokens=k_tokens, embed_scale=embed_scale, include_suppressed=True,
        )

    from .context import hidden_states_at_layers

    prompt = prompt or f"Tell me about {entity}."
    layers = vindex.band(band)
    hs = hidden_states_at_layers(model, tokenizer, prompt, layers, device=device)
    base = (
        hidden_states_at_layers(model, tokenizer, baseline_prompt, layers, device=device)
        if baseline_prompt
        else None
    )
    rows: list[Association] = []
    for L in layers:
        q = hs[L] - base[L] if base is not None else hs[L]
        for f, sim in top_features(vindex, L, q, k=per_layer, include_suppressed=True):
            tok_ids, _ = describe_feature(vindex, L, f, k=k_tokens)
            toks = tokenizer.batch_decode([[int(t)] for t in tok_ids])
            rows.append(
                Association(
                    layer=L, feature=f, sim=float(sim),
                    tokens=[t.strip() for t in toks],
                    suppressed=(L, f) in vindex.suppressed,
                )
            )
    rows.sort(key=lambda r: -r.sim)
    return rows
