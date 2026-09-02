"""Edit + evaluate + describe layer: synthetic Llama-style model, no network."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from marv.edit import ablate, restore, suppress
from marv.evaluate import Probe, diff_battery, run_battery, study_edit
from marv.extract import default_layer_bands, extract
from marv.probe import build_down_meta, describe_entity, describe_feature, logit_lens, top_features


def tiny_model():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


class FakeTok:
    """Word-level tokenizer over a fixed mini vocab (ids < 64), deterministic."""

    _WORDS = [
        "<pad>", "the", "capital", "of", "France", "is", "Paris", "Germany",
        "Berlin", "Italy", "Rome", "language", "French", "German", "weather",
        "in", "Tokyo", "Japan", "a", "and", "to",
    ]

    def __init__(self):
        self.i2w = {i: w for i, w in enumerate(self._WORDS)}
        self.w2i = {w: i for i, w in enumerate(self._WORDS)}

    def _tok(self, w: str) -> int:
        if w in self.w2i:
            return self.w2i[w]
        return (sum(ord(c) for c in w) % 40) + 21

    def encode(self, text, add_special_tokens=False):
        return [self._tok(w) for w in text.replace("?", " ").replace(".", " ").split()]

    def decode(self, ids):
        return " ".join(self.i2w.get(int(i), f"<{int(i)}>") for i in ids)

    def batch_decode(self, seqs):
        return [self.decode(s) for s in seqs]

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}


def test_default_layer_bands_cover_all_layers_contiguously():
    for n in (2, 4, 12, 30, 32):
        b = default_layer_bands(n)
        assert set(b) == {"syntax", "knowledge", "output"}
        assert b["syntax"][0] == 0
        assert b["output"][1] == n - 1
        # no gaps between bands (overlap tolerated for tiny models)
        assert b["knowledge"][0] <= b["syntax"][1] + 1
        assert b["output"][0] <= b["knowledge"][1] + 1
        covered = set()
        for lo, hi in b.values():
            covered |= set(range(lo, hi + 1))
        assert covered == set(range(n))


def test_build_down_meta_matches_uncached_logit_lens():
    vindex = extract(tiny_model())
    build_down_meta(vindex, k=8)
    assert vindex.down_meta_tokens is not None
    for layer in range(vindex.num_layers):
        for feat in (0, 5, 17, 31):
            cached_ids, _ = describe_feature(vindex, layer, feat, k=5)
            fresh_ids, _ = logit_lens(vindex, vindex.down[layer][:, feat], k=5)
            assert list(map(int, cached_ids)) == list(map(int, fresh_ids))


def test_suppressed_hides_feature_from_top_features():
    vindex = extract(tiny_model())
    query = vindex.gate[1][7]  # feature 7 matches itself best
    assert top_features(vindex, 1, query, k=3)[0][0] == 7

    vindex.suppress(1, 7)
    hits = top_features(vindex, 1, query, k=3)
    assert 7 not in [f for f, _ in hits]
    assert top_features(vindex, 1, query, k=3, include_suppressed=True)[0][0] == 7


def test_save_load_roundtrips_new_fields(tmp_path):
    vindex = extract(tiny_model())
    build_down_meta(vindex, k=6)
    vindex.suppress(0, 3)
    vindex.suppress(2, 19)
    path = str(tmp_path / "v.npz")
    vindex.save(path)

    from marv.extract import VindexLite

    loaded = VindexLite.load(path)
    assert loaded.suppressed == {(0, 3), (2, 19)}
    assert loaded.layer_bands == vindex.layer_bands
    assert loaded.down_meta_tokens is not None
    assert loaded.band("knowledge") == vindex.band("knowledge")
    a, _ = describe_feature(loaded, 1, 4, k=5)
    b, _ = logit_lens(loaded, loaded.down[1][:, 4], k=5)
    assert list(map(int, a)) == list(map(int, b))


def test_suppress_context_manager_changes_then_restores_logits():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        base = model(input_ids=ids).logits.clone()

    with suppress(model, [(1, 7), (2, 3)]):
        with torch.no_grad():
            edited = model(input_ids=ids).logits.clone()
    assert not torch.allclose(base, edited)

    with torch.no_grad():
        after = model(input_ids=ids).logits
    assert torch.allclose(base, after)


def test_ablate_and_restore():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        base = model(input_ids=ids).logits.clone()

    saved = ablate(model, [(1, 7)])
    assert torch.count_nonzero(model.model.layers[1].mlp.down_proj.weight[:, 7]) == 0
    with torch.no_grad():
        assert not torch.allclose(base, model(input_ids=ids).logits)

    restore(model, saved)
    with torch.no_grad():
        assert torch.allclose(base, model(input_ids=ids).logits)


def test_run_battery_and_diff_no_edit_is_all_unchanged():
    model, tok = tiny_model(), FakeTok()
    probes = [
        Probe("the capital of France is", "Paris", ("capital", "france")),
        Probe("the capital of Germany is", "Berlin", ("capital", "germany")),
        Probe("the weather in Tokyo is", "a", ("weather",)),
    ]
    r1 = run_battery(model, tok, probes)
    r2 = run_battery(model, tok, probes)
    rep = diff_battery(r1, r2)
    assert all(row.verdict == "unchanged" for row in rep.rows)
    assert "unchanged" in rep.summary()
    out_full = rep.show(full=True)
    assert "the capital of France is" in out_full


def test_study_edit_returns_report():
    model, tok = tiny_model(), FakeTok()
    battery = [
        Probe("the capital of France is", "Paris", ("capital",)),
        Probe("the capital of Italy is", "Rome", ("capital",)),
        Probe("the language of France is", "French", ("language",)),
    ]
    rep = study_edit(model, tok, suppress(model, [(1, 7), (1, 12)]), battery)
    assert len(rep.rows) == 3
    assert all(v in {"flipped", "degraded", "improved", "unchanged"} for v in (r.verdict for r in rep.rows))
    s = rep.show()
    assert isinstance(s, str)


def test_extract_streaming_from_safetensors(tmp_path):
    from safetensors.numpy import save_file

    from marv.extract import extract_streaming

    h, inter, vocab, n = 8, 16, 32, 3
    t = {}
    rng = np.random.default_rng(0)
    for i in range(n):
        t[f"model.layers.{i}.mlp.gate_proj.weight"] = rng.standard_normal((inter, h), dtype=np.float32)
        t[f"model.layers.{i}.mlp.down_proj.weight"] = rng.standard_normal((h, inter), dtype=np.float32)
    t["model.embed_tokens.weight"] = rng.standard_normal((vocab, h), dtype=np.float32)
    t["model.norm.weight"] = np.ones(h, np.float32)
    save_file(t, str(tmp_path / "model.safetensors"))

    v = extract_streaming(str(tmp_path))
    assert v.num_layers == n
    assert v.hidden_size == h
    assert v.gate[0].shape == (inter, h)
    assert v.lm_head is v.embed  # tied (no lm_head key)
    assert v.layer_bands is not None


def test_describe_entity_sorted_and_tokenized():
    vindex = extract(tiny_model())
    build_down_meta(vindex, k=6)
    rows = describe_entity(vindex, FakeTok(), "France", band="knowledge", k_features=3)
    assert rows
    sims = [r.sim for r in rows]
    assert sims == sorted(sims, reverse=True)
    assert all(isinstance(r.tokens, list) and r.tokens for r in rows)

    vindex.suppress(rows[0].layer, rows[0].feature)
    rows2 = describe_entity(vindex, FakeTok(), "France", band="knowledge", k_features=3)
    flagged = [r for r in rows2 if r.suppressed]
    assert any(r.layer == rows[0].layer and r.feature == rows[0].feature for r in flagged)
