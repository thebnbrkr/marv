"""Gate-KNN + logit-lens: browse what an FFN feature does.

Embed a query, do cosine-similarity KNN across a layer's gate rows to find
firing features, then project each hit's down column through norm+lm_head to
read off the tokens it promotes. Plain numpy -- a 135M/1.7B model's gate
matrix per layer is small enough that brute-force cosine similarity is
instant, no KNN index needed.

None of this runs the model. `top_features` reads gate rows, `logit_lens`
reads the unembedding -- static weight math, no forward pass, no attention.
For the contextual version (query = a real hidden state) see marv/context.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import VindexLite


def _l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(norm, 1e-8, None)


def top_features(
    vindex: VindexLite,
    layer: int,
    query: np.ndarray,
    k: int = 10,
    include_suppressed: bool = False,
):
    """Cosine-similarity KNN over one layer's gate rows -- which features fire
    for this residual-space direction. Returns [(feature_idx, cos_sim), ...].

    Features in `vindex.suppressed` are dropped unless `include_suppressed`
    -- that is the whole mechanism behind marv's retrieval-layer
    suppression: hidden here, still present in the weights.
    """
    gate = vindex.gate[layer]
    q = query / max(np.linalg.norm(query), 1e-8)
    sims = _l2_normalize_rows(gate) @ q
    if not include_suppressed and vindex.suppressed:
        drop = [f for (l, f) in vindex.suppressed if l == layer]
        if drop:
            sims = sims.copy()
            sims[drop] = -np.inf
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


def build_down_meta(vindex: VindexLite, k: int = 12, col_chunk: int = 4096, device: str | None = None) -> VindexLite:
    """Precompute every feature's top-k promoted tokens once (logit lens over
    every down column) and cache them on the vindex, so describe_feature()
    becomes a lookup instead of a (vocab x hidden) matmul per call.

    This is exactly LARQL's `down_meta.bin`. Pure weight math -- one
    `lm_head @ down[layer]` per layer, no model execution. Mutates and
    returns `vindex`.

    `device`: pass "cuda" to run the matmul + top-k on the GPU (torch). On a
    T4 this is ~50x faster than numpy for a 150k-vocab model -- worth it, and
    the vindex stays on CPU. Default (None) uses numpy; pass "cpu" to force
    the torch CPU path.
    """
    if device is not None:
        return _build_down_meta_torch(vindex, k, col_chunk, device)

    tok_cache: list[np.ndarray] = []
    score_cache: list[np.ndarray] = []
    w = vindex.lm_head.astype(np.float32)  # (vocab, hidden)
    fnw = vindex.final_norm_weight
    for layer in range(vindex.num_layers):
        dn = vindex.down[layer].astype(np.float32)  # (hidden, intermediate)
        rms = np.sqrt((dn**2).mean(axis=0, keepdims=True) + vindex.norm_eps)
        v = dn / rms
        if fnw is not None:
            v = v * fnw[:, None]
        inter = v.shape[1]
        kk = min(k, w.shape[0])
        toks = np.zeros((inter, kk), dtype=np.int32)
        scores = np.zeros((inter, kk), dtype=np.float32)
        for c0 in range(0, inter, col_chunk):
            c1 = min(c0 + col_chunk, inter)
            logits = w @ v[:, c0:c1]  # (vocab, chunk)
            part = np.argpartition(-logits, kth=kk - 1, axis=0)[:kk]  # (kk, chunk)
            cols = np.arange(c1 - c0)
            part_scores = logits[part, cols]
            order = np.argsort(-part_scores, axis=0)
            part = np.take_along_axis(part, order, axis=0)
            part_scores = np.take_along_axis(part_scores, order, axis=0)
            toks[c0:c1] = part.T
            scores[c0:c1] = part_scores.T
        tok_cache.append(toks)
        score_cache.append(scores)
    vindex.down_meta_tokens = tok_cache
    vindex.down_meta_scores = score_cache
    return vindex


def _build_down_meta_torch(vindex: VindexLite, k: int, col_chunk: int, device: str) -> VindexLite:
    import torch

    w = torch.as_tensor(vindex.lm_head, dtype=torch.float32, device=device)  # (vocab, hidden)
    fnw = None if vindex.final_norm_weight is None else torch.as_tensor(
        vindex.final_norm_weight, dtype=torch.float32, device=device
    )
    kk = min(k, w.shape[0])
    tok_cache, score_cache = [], []
    with torch.no_grad():
        for layer in range(vindex.num_layers):
            dn = torch.as_tensor(vindex.down[layer], dtype=torch.float32, device=device)
            v = dn / torch.sqrt((dn**2).mean(dim=0, keepdim=True) + vindex.norm_eps)
            if fnw is not None:
                v = v * fnw[:, None]
            inter = v.shape[1]
            toks = torch.zeros((inter, kk), dtype=torch.int32)
            scores = torch.zeros((inter, kk), dtype=torch.float32)
            for c0 in range(0, inter, col_chunk):
                c1 = min(c0 + col_chunk, inter)
                logits = w @ v[:, c0:c1]  # (vocab, chunk)
                s, idx = torch.topk(logits, kk, dim=0)
                toks[c0:c1] = idx.T.to(torch.int32).cpu()
                scores[c0:c1] = s.T.float().cpu()
            tok_cache.append(toks.numpy())
            score_cache.append(scores.numpy())
    vindex.down_meta_tokens = tok_cache
    vindex.down_meta_scores = score_cache
    return vindex


def describe_feature(vindex: VindexLite, layer: int, feature_idx: int, k: int = 5):
    """One FFN feature -> its top-k promoted tokens (e.g. a
    `capital -> Paris` association), read straight off the down_proj column.
    Uses the build_down_meta() cache when it covers `k`."""
    cache = vindex.down_meta_tokens
    if cache is not None and k <= cache[layer].shape[1]:
        return (
            cache[layer][feature_idx][:k].copy(),
            vindex.down_meta_scores[layer][feature_idx][:k].copy(),
        )
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


@dataclass
class Association:
    """One row of describe_entity(): a firing feature and what it promotes."""

    layer: int
    feature: int
    sim: float
    tokens: list[str]
    suppressed: bool = False

    def __repr__(self) -> str:
        s = " (suppressed)" if self.suppressed else ""
        return f"L{self.layer} f{self.feature} sim={self.sim:.2f} -> {self.tokens}{s}"


def describe_entity(
    vindex: VindexLite,
    tokenizer,
    entity: str,
    *,
    band: str = "knowledge",
    layers: list[int] | None = None,
    k_features: int = 6,
    k_tokens: int = 5,
    embed_scale: float = 1.0,
    include_suppressed: bool = True,
):
    """Graph-free DESCRIBE: embed `entity` (averaged over its tokens), KNN its
    gate features across the knowledge band, label each by its promoted
    tokens. Returns a list of `Association`, sorted by similarity.

    The query is the *bare* embedding (no forward pass), so matches are
    weaker than context.describe_prompt()'s contextual version -- but it
    needs only the vindex + tokenizer. Suppressed features are still listed
    (flagged) so you can see the constellation you carved into.
    """
    ids = tokenizer.encode(entity, add_special_tokens=False)
    if not ids:
        raise ValueError(f"{entity!r} tokenized to nothing")
    q = vindex.embed[ids].mean(axis=0).astype(np.float32) * embed_scale
    layers = layers if layers is not None else vindex.band(band)
    rows: list[Association] = []
    for layer in layers:
        hits = top_features(vindex, layer, q, k=k_features, include_suppressed=include_suppressed)
        for f, sim in hits:
            tok_ids, _ = describe_feature(vindex, layer, f, k=k_tokens)
            toks = tokenizer.batch_decode([[int(t)] for t in tok_ids])
            rows.append(
                Association(
                    layer=layer,
                    feature=f,
                    sim=float(sim),
                    tokens=[t.strip() for t in toks],
                    suppressed=(layer, f) in vindex.suppressed,
                )
            )
    rows.sort(key=lambda r: -r.sim)
    return rows
