"""Feature-level diff between two checkpoints -- MARV's version-control
primitive.

Where a LoRA delta or a full state_dict diff tells you "these matrices
changed," this tells you *which individual FFN features* moved, and by how
much -- a sparse, inspectable list instead of a dense weight delta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import VindexLite


@dataclass
class FeatureDelta:
    layer: int
    feature_idx: int
    gate_cos_sim: float  # 1.0 = unchanged direction, -1.0 = flipped
    down_cos_sim: float
    gate_norm_ratio: float  # ||gate_after|| / ||gate_before||


def diff(base: VindexLite, tuned: VindexLite) -> list[FeatureDelta]:
    """Per-(layer, feature) cosine similarity of the gate row and down column
    between two checkpoints sharing the same architecture. Low gate_cos_sim
    means fine-tuning repointed what that feature fires on; low down_cos_sim
    means it repointed what the feature promotes when it fires.
    """
    if base.num_layers != tuned.num_layers:
        raise ValueError(
            "marv.diff requires matching layer counts (same base architecture); "
            f"got {base.num_layers} vs {tuned.num_layers}"
        )
    deltas: list[FeatureDelta] = []
    for layer in range(base.num_layers):
        g0, g1 = base.gate[layer], tuned.gate[layer]
        d0, d1 = base.down[layer], tuned.down[layer]
        if g0.shape != g1.shape:
            raise ValueError(f"layer {layer}: gate shape {g0.shape} != {g1.shape}")

        n0 = np.linalg.norm(g0, axis=1)
        n1 = np.linalg.norm(g1, axis=1)
        gate_cos = np.sum(g0 * g1, axis=1) / (n0 * n1 + 1e-8)

        down_cos = np.sum(d0 * d1, axis=0) / (
            np.linalg.norm(d0, axis=0) * np.linalg.norm(d1, axis=0) + 1e-8
        )
        ratio = n1 / (n0 + 1e-8)

        for f in range(g0.shape[0]):
            deltas.append(
                FeatureDelta(
                    layer=layer,
                    feature_idx=f,
                    gate_cos_sim=float(gate_cos[f]),
                    down_cos_sim=float(down_cos[f]),
                    gate_norm_ratio=float(ratio[f]),
                )
            )
    return deltas


def most_changed(deltas: list[FeatureDelta], k: int = 20) -> list[FeatureDelta]:
    """Rank by how much the feature's *input direction* moved -- a small,
    portable record of exactly which neurons fine-tuning actually touched."""
    return sorted(deltas, key=lambda d: d.gate_cos_sim)[:k]


def per_layer_score(deltas: list[FeatureDelta], metric: str = "max") -> dict[int, float]:
    """Collapse per-feature deltas to one weight-change score per layer, so a
    layer ranking from `diff()` can be compared against a layer ranking from
    something else (e.g. marv.layer_heatmap's per-prompt activation
    divergence) instead of eyeballing which layers show up in a top-k list.

    Most features in any layer barely move (gate_cos_sim ~0.999+), so a plain
    per-layer mean would drown out the handful that actually changed --
    `metric="max"` (default) reports the single most-changed feature's score
    per layer, matching what `most_changed()` already surfaces. `"mean_topk"`
    averages the top 5 changed features per layer instead, for a slightly
    less single-outlier-sensitive summary.
    """
    by_layer: dict[int, list[float]] = {}
    for d in deltas:
        by_layer.setdefault(d.layer, []).append(1.0 - d.gate_cos_sim)

    scores: dict[int, float] = {}
    for layer, changes in by_layer.items():
        if metric == "max":
            scores[layer] = max(changes)
        elif metric == "mean_topk":
            top = sorted(changes, reverse=True)[:5]
            scores[layer] = sum(top) / len(top)
        else:
            raise ValueError(f"unknown metric {metric!r}, expected 'max' or 'mean_topk'")
    return scores
