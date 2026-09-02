"""Feature-activation matrices across many probe words/categories -- the data
behind a "does this model have distinct factual/instruction/coding features,
or do the same few neurons fire for everything" heatmap.

Built on marv.context.describe_prompt's contextual, baseline-differenced
query (see that module's docstring for why a raw embedding or an
un-differenced hidden state both give weak, undifferentiated signal). This
module answers a different question than describe_prompt though:
describe_prompt asks "what does this one word fire and promote," this module
asks "across many words spanning several concepts, which features are
specific to one concept and which fire for several" (polysemanticity).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .extract import VindexLite
from .probe import top_features
from .context import hidden_states_at_layers


def _differenced_query(model, tokenizer, word: str, layer: int, template: str, baseline_prompt: str, device: str):
    prompt = template.format(word=word)
    h = hidden_states_at_layers(model, tokenizer, prompt, [layer], device=device)[layer]
    if baseline_prompt is not None:
        b = hidden_states_at_layers(model, tokenizer, baseline_prompt, [layer], device=device)[layer]
        h = h - b
    return h


@dataclass
class ActivationMatrix:
    words: list[str]  # one entry per probed word
    categories: list[str]  # categories[i] is words[i]'s category
    feature_ids: list[int]  # column labels: (layer-local) feature indices
    layer: int
    matrix: np.ndarray  # (len(words), len(feature_ids)) cosine similarities


def activation_matrix(
    vindex: VindexLite,
    model,
    tokenizer,
    categories: dict[str, list[str]],
    layer: int,
    top_k_per_word: int = 15,
    template: str = "I want to talk about {word}",
    baseline_prompt: str | None = "I want to talk about",
    device: str = "cpu",
) -> ActivationMatrix:
    """For each word in each category, find its top firing features at
    `layer`; the heatmap's columns are the *union* of those hits across all
    words (keeps the plot to a legible width instead of the full
    intermediate_size). Cells are cosine similarity, so a column with high
    values across rows from different categories is a polysemantic feature;
    a column high only within one category's rows is concept-specific.
    """
    words = [w for ws in categories.values() for w in ws]
    word_categories = [cat for cat, ws in categories.items() for _ in ws]
    queries = {w: _differenced_query(model, tokenizer, w, layer, template, baseline_prompt, device) for w in words}

    feature_set: set[int] = set()
    for w in words:
        for feature_idx, _ in top_features(vindex, layer, queries[w], k=top_k_per_word):
            feature_set.add(feature_idx)
    feature_ids = sorted(feature_set)

    gate = vindex.gate[layer][feature_ids]  # (num_selected_features, hidden_size)
    gate_norm = gate / np.clip(np.linalg.norm(gate, axis=1, keepdims=True), 1e-8, None)

    matrix = np.zeros((len(words), len(feature_ids)), dtype=np.float32)
    for i, w in enumerate(words):
        q = queries[w]
        q = q / max(np.linalg.norm(q), 1e-8)
        matrix[i] = gate_norm @ q

    return ActivationMatrix(words=words, categories=word_categories, feature_ids=feature_ids, layer=layer, matrix=matrix)


def polysemantic_features(am: ActivationMatrix, threshold: float = 0.15, min_categories: int = 2) -> dict[int, set[str]]:
    """Feature columns that fire (>= threshold) for words spanning at least
    `min_categories` distinct categories -- concrete evidence a feature is
    polysemantic, as opposed to specific to one concept. Returns
    {feature_id: {category, ...}}."""
    cats_per_feature: dict[int, set[str]] = defaultdict(set)
    for i, cat in enumerate(am.categories):
        for j, feature_id in enumerate(am.feature_ids):
            if am.matrix[i, j] >= threshold:
                cats_per_feature[feature_id].add(cat)
    return {fid: cats for fid, cats in cats_per_feature.items() if len(cats) >= min_categories}


def plot_heatmap(am: ActivationMatrix, title: str = ""):
    """Matplotlib heatmap: words on the y-axis (grouped/colored by category),
    features on the x-axis. Requires matplotlib (`pip install matplotlib`,
    already present in Colab)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(am.feature_ids) * 0.4), max(4, len(am.words) * 0.35)))
    im = ax.imshow(am.matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(am.feature_ids)))
    ax.set_xticklabels([f"f{i}" for i in am.feature_ids], rotation=90, fontsize=7)
    ax.set_yticks(range(len(am.words)))
    ax.set_yticklabels([f"{w} ({c})" for w, c in zip(am.words, am.categories)], fontsize=8)
    ax.set_xlabel(f"features at layer {am.layer}")
    ax.set_title(title or f"feature activation, layer {am.layer}")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    return fig
