"""Layer x feature activation heatmaps for a single prompt -- "which features
fire, and at which depth" for one input, as opposed to marv.heatmap's "which
features distinguish these categories of words."

marv.heatmap.activation_matrix reduces each word to its top-k firing
features to keep the plot legible across many words. This module instead
keeps every feature at every layer for one prompt -- a dense
(num_layers, intermediate_size) grid, which is the shape people actually
mean by "feature activation heatmap."

A caveat worth stating plainly, because it's easy to assume otherwise: MARV's
"features" are raw MLP neurons -- one gate row per layer -- not a shared
cross-layer feature dictionary (as in sparse-autoencoder interpretability
work). Feature index 134 at layer 3 and feature index 134 at layer 18 are
unrelated neurons that happen to share a column position in this grid; there
is no persistent "feature 134" threading through the whole model.
`peak_activation_trace()` is the honest version of "how does a signal evolve
across layers" for this: it tracks the *strongest* matching feature at each
layer, not a fixed column index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import VindexLite
from .toolcall import hidden_states_at_layers


@dataclass
class LayerFeatureHeatmap:
    prompt: str
    layers: list[int]
    matrix: np.ndarray  # (len(layers), intermediate_size) cosine similarities


def compute(
    vindex: VindexLite,
    model,
    tokenizer,
    prompt: str,
    layers: list[int] | None = None,
    baseline_prompt: str | None = None,
    device: str = "cpu",
) -> LayerFeatureHeatmap:
    """One row per layer: cosine similarity of that layer's (optionally
    baseline-differenced -- see marv.toolcall.describe_prompt for why) hidden
    state against every one of that layer's gate rows. Defaults to every
    layer in the model.
    """
    layers = list(layers) if layers is not None else list(range(vindex.num_layers))
    hidden = hidden_states_at_layers(model, tokenizer, prompt, layers, device=device)
    baseline = (
        hidden_states_at_layers(model, tokenizer, baseline_prompt, layers, device=device)
        if baseline_prompt is not None
        else None
    )
    rows = []
    for layer in layers:
        query = hidden[layer] - baseline[layer] if baseline is not None else hidden[layer]
        query = query / max(np.linalg.norm(query), 1e-8)
        gate = vindex.gate[layer]
        gate_norm = gate / np.clip(np.linalg.norm(gate, axis=1, keepdims=True), 1e-8, None)
        rows.append(gate_norm @ query)
    return LayerFeatureHeatmap(prompt=prompt, layers=layers, matrix=np.stack(rows))


def difference(a: LayerFeatureHeatmap, b: LayerFeatureHeatmap) -> LayerFeatureHeatmap:
    """a - b over the same (layers, feature-index) grid -- e.g. a tool prompt's
    heatmap minus a non-tool phrasing's: positive cells are features that
    fire more for `a`."""
    if a.matrix.shape != b.matrix.shape or a.layers != b.layers:
        raise ValueError("difference() needs two heatmaps computed over the same layers")
    return LayerFeatureHeatmap(prompt=f"{a.prompt!r} minus {b.prompt!r}", layers=a.layers, matrix=a.matrix - b.matrix)


def peak_activation_trace(hm: LayerFeatureHeatmap) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer strongest activation and which feature produced it -- see
    the module docstring for why this, not a fixed feature index, is the
    honest way to trace a signal's strength across layers. Returns
    (peak_values, peak_feature_ids), each length len(hm.layers)."""
    return hm.matrix.max(axis=1), hm.matrix.argmax(axis=1)


def layer_trace(hm: LayerFeatureHeatmap, feature_index: int) -> np.ndarray:
    """Raw column slice: activation of column `feature_index` across every
    layer. Only meaningful as 'the same neuron across layers' by coincidence
    (see module docstring) -- typically more useful for reading off the
    column `peak_activation_trace` names at one layer of interest and
    checking whether it happens to stay active nearby."""
    return hm.matrix[:, feature_index]


def plot(hm: LayerFeatureHeatmap, title: str = "", cmap: str = "viridis", vmin=None, vmax=None):
    """Matplotlib heatmap: feature index on x, layer on y, color = cosine
    similarity. Requires matplotlib (already present in Colab)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, max(4, len(hm.layers) * 0.28)))
    im = ax.imshow(hm.matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_yticks(range(len(hm.layers)))
    ax.set_yticklabels(hm.layers, fontsize=7)
    ax.set_xlabel("feature index")
    ax.set_ylabel("layer")
    ax.set_title(title or f"activation heatmap: {hm.prompt!r}")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    return fig
