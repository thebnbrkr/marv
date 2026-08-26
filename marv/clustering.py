"""Cluster prompts by their FFN feature-activation pattern at one layer.

Answers a different question than marv.heatmap (which words distinguish
categories at one layer, reduced to the top-k union for legibility) and
marv.layer_heatmap (every feature across every layer for one prompt): given
several *prompts* grouped into behaviors (e.g. several tool-call phrasings vs
several plain requests), do they group together in feature space, and which
specific features are consistently responsible for that grouping ("tool-
cluster features")?

Requires scikit-learn for PCA/t-SNE (`pip install scikit-learn`, already
present in Colab).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import VindexLite
from .layer_heatmap import compute as layer_heatmap_compute


@dataclass
class PromptActivations:
    prompts: list[str]
    groups: list[str]  # groups[i] is the group label for prompts[i]
    layer: int
    matrix: np.ndarray  # (num_prompts, intermediate_size) cosine similarities


def prompt_activations(
    vindex: VindexLite,
    model,
    tokenizer,
    groups: dict[str, list[str]],
    layer: int,
    baseline_prompt: str | None = None,
    device: str = "cpu",
) -> PromptActivations:
    """One row per prompt: that prompt's full dense feature-activation vector
    at `layer` (reuses marv.layer_heatmap.compute, restricted to one layer).
    Unlike marv.heatmap, there's no top-k column reduction here -- PCA/t-SNE
    can operate directly on the full intermediate_size-dimensional vector;
    the reduction in marv.heatmap exists only to keep an imshow plot legible.
    """
    prompts = [p for ps in groups.values() for p in ps]
    labels = [g for g, ps in groups.items() for _ in ps]
    rows = []
    for prompt in prompts:
        hm = layer_heatmap_compute(vindex, model, tokenizer, prompt, layers=[layer], baseline_prompt=baseline_prompt, device=device)
        rows.append(hm.matrix[0])
    return PromptActivations(prompts=prompts, groups=labels, layer=layer, matrix=np.stack(rows))


def reduce_pca(pa: PromptActivations, n_components: int = 2) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components).fit_transform(pa.matrix)


def reduce_tsne(pa: PromptActivations, n_components: int = 2, perplexity: float | None = None, random_state: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = pa.matrix.shape[0]
    if perplexity is None:
        # sklearn requires perplexity < n_samples; t-SNE also isn't
        # meaningful on a handful of points, but this keeps it from
        # hard-erroring on a small prompt set.
        perplexity = max(2, min(30, n - 1))
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init="pca").fit_transform(pa.matrix)


def plot_projection(pa: PromptActivations, points: np.ndarray, title: str = ""):
    """Scatter plot of a 2D projection (from reduce_pca/reduce_tsne), colored
    by group, with each prompt's text annotated."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for group in sorted(set(pa.groups)):
        idx = [i for i, g in enumerate(pa.groups) if g == group]
        ax.scatter(points[idx, 0], points[idx, 1], label=group, s=70)
    for i, prompt in enumerate(pa.prompts):
        label = prompt if len(prompt) <= 28 else prompt[:25] + "..."
        ax.annotate(label, (points[i, 0], points[i, 1]), fontsize=7, alpha=0.75, xytext=(4, 4), textcoords="offset points")
    ax.legend()
    ax.set_title(title or f"prompt clustering, layer {pa.layer}")
    fig.tight_layout()
    return fig


def cluster_features(
    pa: PromptActivations,
    group: str,
    min_group_activation: float = 0.2,
    other_threshold: float = 0.1,
    top_n: int = 20,
) -> list[int]:
    """Features consistently high (mean >= min_group_activation) within
    `group`'s prompts and low (mean < other_threshold) across every other
    group -- the literal feature-level definition of a "tool-cluster" (or
    whatever `group` is), rather than just the geometric separation a PCA
    plot shows. Ranked by how strongly they fire within the group.
    """
    idx_in = [i for i, g in enumerate(pa.groups) if g == group]
    idx_out = [i for i, g in enumerate(pa.groups) if g != group]
    if not idx_in:
        raise ValueError(f"no prompts found for group {group!r}")

    mean_in = pa.matrix[idx_in].mean(axis=0)
    mean_out = pa.matrix[idx_out].mean(axis=0) if idx_out else np.full_like(mean_in, -np.inf)

    candidates = np.where((mean_in >= min_group_activation) & (mean_out < other_threshold))[0]
    ranked = candidates[np.argsort(-mean_in[candidates])]
    return ranked[:top_n].tolist()
