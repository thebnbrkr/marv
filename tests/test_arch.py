"""Adapter + extraction sanity checks on a tiny synthetic Llama-style model
-- no network access, no real checkpoint download, so this runs anywhere."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from marv.arch import LlamaStyleFFN, detect_adapter
from marv.clustering import PromptActivations, cluster_features
from marv.diff import FeatureDelta, diff, most_changed, per_layer_score
from marv.extract import extract
from marv.heatmap import ActivationMatrix, polysemantic_features
from marv.layer_heatmap import (
    LayerFeatureHeatmap,
    difference,
    layer_attribution,
    layer_trace,
    peak_activation_trace,
    top_features_per_layer,
)
from marv.probe import describe_feature, logit_lens, top_features


def tiny_model():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def test_detect_adapter_returns_llama_style():
    model = tiny_model()
    assert isinstance(detect_adapter(model), LlamaStyleFFN)


def test_extract_shapes():
    model = tiny_model()
    vindex = extract(model)
    assert vindex.num_layers == 2
    assert vindex.hidden_size == 16
    for layer in range(vindex.num_layers):
        assert vindex.gate[layer].shape == (32, 16)
        assert vindex.down[layer].shape == (16, 32)
    assert vindex.lm_head.shape == (64, 16)


def test_probe_runs_and_shapes_match():
    model = tiny_model()
    vindex = extract(model)
    query = vindex.gate[0][3]  # a real gate row should match itself best
    hits = top_features(vindex, layer=0, query=query, k=5)
    assert hits[0][0] == 3
    assert hits[0][1] > 0.99

    tok_ids, logits = describe_feature(vindex, layer=0, feature_idx=3, k=4)
    assert tok_ids.shape == (4,)
    assert logits.shape == (4,)

    idx, logits = logit_lens(vindex, np.zeros(16, dtype=np.float32), k=4)
    assert idx.shape == (4,)


def test_diff_identical_model_is_all_ones():
    model = tiny_model()
    vindex = extract(model)
    deltas = diff(vindex, vindex)
    assert all(abs(d.gate_cos_sim - 1.0) < 1e-5 for d in deltas)
    assert most_changed(deltas, k=3)[0].gate_cos_sim <= 1.0 + 1e-5


def test_diff_detects_perturbed_feature():
    model = tiny_model()
    vindex_base = extract(model)
    vindex_tuned = extract(model)
    vindex_tuned.gate[1][7] = -vindex_tuned.gate[1][7]  # flip one feature's direction

    deltas = diff(vindex_base, vindex_tuned)
    worst = most_changed(deltas, k=1)[0]
    assert worst.layer == 1
    assert worst.feature_idx == 7
    assert worst.gate_cos_sim < -0.99


def test_polysemantic_features_flags_shared_columns():
    # 3 words x 4 features. f0 fires for both categories (polysemantic);
    # f1 only for "factual"; f2/f3 are noise below threshold everywhere.
    am = ActivationMatrix(
        words=["paris", "capital", "summarize"],
        categories=["factual", "factual", "instruction"],
        feature_ids=[0, 1, 2, 3],
        layer=5,
        matrix=np.array(
            [
                [0.30, 0.25, 0.05, 0.02],
                [0.28, 0.22, 0.04, 0.03],
                [0.31, 0.05, 0.03, 0.01],
            ],
            dtype=np.float32,
        ),
    )
    poly = polysemantic_features(am, threshold=0.15, min_categories=2)
    assert set(poly.keys()) == {0}
    assert poly[0] == {"factual", "instruction"}


def test_peak_activation_trace_and_layer_trace():
    # 3 layers x 4 features: layer 1's peak is feature 2, others are flat.
    hm = LayerFeatureHeatmap(
        prompt="test",
        layers=[0, 1, 2],
        matrix=np.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1],
                [0.2, 0.2, 0.2, 0.2],
            ],
            dtype=np.float32,
        ),
    )
    peak_values, peak_features = peak_activation_trace(hm)
    np.testing.assert_allclose(peak_values, [0.1, 0.9, 0.2])
    assert list(peak_features) == [0, 2, 0]  # argmax ties resolve to the first index
    np.testing.assert_allclose(layer_trace(hm, 2), [0.1, 0.9, 0.2])


def test_layer_heatmap_difference():
    a = LayerFeatureHeatmap(prompt="a", layers=[0, 1], matrix=np.array([[0.5, 0.2], [0.3, 0.1]], dtype=np.float32))
    b = LayerFeatureHeatmap(prompt="b", layers=[0, 1], matrix=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    d = difference(a, b)
    np.testing.assert_allclose(d.matrix, [[0.4, 0.0], [0.0, -0.3]])


def test_per_layer_score_max_and_mean_topk():
    deltas = [
        FeatureDelta(layer=0, feature_idx=0, gate_cos_sim=0.999, down_cos_sim=0.999, gate_norm_ratio=1.0),
        FeatureDelta(layer=0, feature_idx=1, gate_cos_sim=0.5, down_cos_sim=0.999, gate_norm_ratio=1.0),
        FeatureDelta(layer=1, feature_idx=0, gate_cos_sim=0.999, down_cos_sim=0.999, gate_norm_ratio=1.0),
    ]
    scores = per_layer_score(deltas, metric="max")
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(0.001)

    mean_scores = per_layer_score(deltas, metric="mean_topk")
    # layer 0 has only 2 features, both included in the top-5 average
    assert mean_scores[0] == pytest.approx((0.001 + 0.5) / 2)


def test_top_features_per_layer_and_layer_attribution():
    hm = LayerFeatureHeatmap(
        prompt="test",
        layers=[0, 1],
        matrix=np.array([[0.1, 0.9, 0.3], [0.4, 0.2, 0.05]], dtype=np.float32),
    )
    top = top_features_per_layer(hm, n=2)
    assert top[0] == [(1, pytest.approx(0.9)), (2, pytest.approx(0.3))]
    assert top[1] == [(0, pytest.approx(0.4)), (1, pytest.approx(0.2))]

    l2 = layer_attribution(hm, mode="l2")
    np.testing.assert_allclose(l2, np.sqrt((hm.matrix**2).sum(axis=1)))
    sum_abs = layer_attribution(hm, mode="sum_abs")
    np.testing.assert_allclose(sum_abs, np.abs(hm.matrix).sum(axis=1))


def test_cluster_features_finds_group_specific_columns():
    # f0 fires for "tool" prompts only; f1 fires for everything (not a
    # cluster feature); f2 is noise everywhere.
    pa = PromptActivations(
        prompts=["call the tool", "use the function", "what is this", "tell me a fact"],
        groups=["tool", "tool", "plain", "plain"],
        layer=10,
        matrix=np.array(
            [
                [0.30, 0.25, 0.02],
                [0.28, 0.22, 0.01],
                [0.02, 0.24, 0.03],
                [0.01, 0.20, 0.02],
            ],
            dtype=np.float32,
        ),
    )
    tool_features = cluster_features(pa, "tool", min_group_activation=0.2, other_threshold=0.1, top_n=5)
    assert tool_features == [0]
