"""Gate-KNN + logit-lens: browse what an FFN feature does.

Embed a query, do cosine-similarity KNN across a layer's gate rows to find
firing features, then project each hit's down column through norm+lm_head to
read off the tokens it promotes. Plain numpy -- a 135M/1.7B model's gate
matrix per layer is small enough that brute-force cosine similarity is
instant, no KNN index needed.
"""
from __future__ import annotations

import numpy as np

from .extract import VindexLite


def _l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(norm, 1e-8, None)


def top_features(vindex: VindexLite, layer: int, query: np.ndarray, k: int = 10):
    """Cosine-similarity KNN over one layer's gate rows -- which features fire
    for this residual-space direction. Returns [(feature_idx, cos_sim), ...]."""
    gate = vindex.gate[layer]
    q = query / max(np.linalg.norm(query), 1e-8)
    sims = _l2_normalize_rows(gate) @ q
    idx = np.argsort(-sims)[:k]
    return list(zip(idx.tolist(), sims[idx].tolist()))


def logit_lens(vindex: VindexLite, vector: np.ndarray, k: int = 10, rms_norm: bool = True):
    """Project a residual-space vector through (RMSNorm +) lm_head -- 'what
    does this direction mean in vocab space.' Returns (top_k_token_ids,
    their_logits).

    Applies the model's *actual* learned final-norm gain (per-channel
    weight), not just a scalar RMS normalization -- skipping that gain means
    the projection is dominated by whichever raw hidden dims happen to have
    large magnitude, rather than by what the model's own norm layer actually
    passes through to the unembedding. Without `final_norm_weight` (e.g. an
    older saved vindex) this falls back to plain unit-RMS scaling.
    """
    v = vector
    if rms_norm:
        v = v / (np.sqrt((v**2).mean() + vindex.norm_eps))
        if vindex.final_norm_weight is not None:
            v = v * vindex.final_norm_weight
    logits = vindex.lm_head @ v
    idx = np.argsort(-logits)[:k]
    return idx, logits[idx]


def describe_feature(vindex: VindexLite, layer: int, feature_idx: int, k: int = 5):
    """One FFN feature -> its top-k promoted tokens (e.g. a
    `capital -> Paris` association), read straight off the down_proj
    column."""
    down_col = vindex.down[layer][:, feature_idx]
    return logit_lens(vindex, down_col, k=k)


def describe(vindex: VindexLite, query: np.ndarray, layers: list[int], k_features: int = 10, k_tokens: int = 5):
    """Full describe: for each layer, find the top firing features for `query`
    and label each one via its promoted tokens. Returns
    {layer: [(feature_idx, cos_sim, top_token_ids, top_logits), ...]}."""
    out = {}
    for layer in layers:
        hits = []
        for feature_idx, sim in top_features(vindex, layer, query, k=k_features):
            tok_ids, logits = describe_feature(vindex, layer, feature_idx, k=k_tokens)
            hits.append((feature_idx, sim, tok_ids, logits))
        out[layer] = hits
    return out
