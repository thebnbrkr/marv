"""Tracing + diagnostics (ported from marv-hyena): synthetic Llama, no network.
Each test checks a tool against an exact answer, not just that it runs."""
from __future__ import annotations

import numpy as np
import torch

from marv.diagnostics import dead_features, find_bottlenecks, health, load_bearing, null_model, write_norms
from marv.edit import suppress
from marv.evaluate import Probe, study_edit
from marv.trace import (
    all_components,
    capture_writes,
    decompose_logit,
    logit_diff_metric,
    mean_ablate,
    mean_writes,
    patch_sweep,
    replace_outputs,
    trace_by_depth,
)
from test_edit import FakeTok, tiny_model

TEXTS = ["the capital of France is Paris", "the capital of Italy is Rome and the weather in Tokyo"]


def test_writes_add_up_to_the_final_residual():
    m, tok = tiny_model(), FakeTok()
    w = capture_writes(m, tok, "the capital of France is", positions=[0, 2, 4])
    assert len(w.parts) == 2 * 3
    assert w.reconstruction_error() < 1e-5


def test_prompt_may_be_a_dict_of_model_inputs():
    m, tok = tiny_model(), FakeTok()
    ids = tok("the capital of France is")["input_ids"]
    a = capture_writes(m, tok, "the capital of France is")
    b = capture_writes(m, tok, {"input_ids": ids})
    torch.testing.assert_close(a.final, b.final)
    d = decompose_logit(m, tok, {"input_ids": ids}, "Paris", "Rome")
    assert d.check_error < 1e-4


def test_direct_attribution_sums_to_the_real_logit_difference():
    m, tok = tiny_model(), FakeTok()
    for target, baseline in (("Paris", None), ("Paris", "Rome"), ("Berlin", "the")):
        d = decompose_logit(m, tok, "the capital of France is", target, baseline)
        assert d.check_error < 1e-4, d.check_error
        assert set(d.by_part()) == {"embed", "attn", "mlp"}


def test_mean_ablation_changes_output_and_cleans_up():
    m, tok = tiny_model(), FakeTok()
    ids = tok("the capital of France is")["input_ids"]
    before = m(input_ids=ids).logits.clone()
    means = mean_writes(m, tok, TEXTS, [(1, "mlp"), (2, "attn")])
    with mean_ablate(m, means):
        during = m(input_ids=ids).logits
    after = m(input_ids=ids).logits
    assert torch.equal(before, after) and not torch.allclose(before, during)


def test_trace_by_depth_starts_at_exactly_one():
    """Prompts differing only at one position: patching the embedding there IS the source run."""
    m, tok = tiny_model(), FakeTok()
    metric = logit_diff_metric(tok, "Rome", "Paris")
    rows = trace_by_depth(m, tok, "the capital of Italy is", "the capital of France is", position=3, metric=metric)
    assert rows[0]["depth"] == -1 and abs(rows[0]["fraction"] - 1.0) < 1e-5
    assert len(rows) == 1 + 3


def test_patching_every_write_reproduces_the_source_prediction():
    """The prompts differ only at position 3, so the last position's embedding is identical.
    Its final residual is embed + all writes, so swapping in EVERY source write must reproduce
    the source run's prediction exactly (fraction 1)."""
    m, tok = tiny_model(), FakeTok()
    metric = logit_diff_metric(tok, "Rome", "Paris")
    src, tgt = "the capital of Italy is", "the capital of France is"
    sweep = patch_sweep(m, tok, src, tgt, metric)
    assert len(sweep.results) == 2 * 3 + 2 and all(np.isfinite(r.fraction) for r in sweep.results)
    cache = {}
    hooks = [m.model.layers[l].self_attn.register_forward_hook(
                 lambda _m, _i, o, l=l: cache.__setitem__((l, "attn"), o[0].detach().clone()))
             for l in range(3)]
    hooks += [m.model.layers[l].mlp.register_forward_hook(
                 lambda _m, _i, o, l=l: cache.__setitem__((l, "mlp"), o.detach().clone()))
              for l in range(3)]
    m(input_ids=tok(src)["input_ids"])
    for h in hooks:
        h.remove()
    with replace_outputs(m, {c: (lambda v: (lambda x: v))(v) for c, v in cache.items()}):
        patched = metric(m(input_ids=tok(tgt)["input_ids"]).logits)
    assert abs(patched - sweep.source_metric) < 1e-4


def test_bottleneck_detector_finds_a_dominating_write():
    m, tok = tiny_model(), FakeTok()
    with replace_outputs(m, {(1, "mlp"): lambda x: x * 1e4}):
        assert find_bottlenecks(m, tok, "the capital of France is") == [(1, "mlp")]
        shares = {(r.layer, r.part): r.share for r in write_norms(m, tok, "the capital of France is")}
        assert shares[(1, "mlp")] > 0.99


def test_load_bearing_flags_a_layer_that_breaks_the_model():
    m, tok = tiny_model(), FakeTok()
    rows = load_bearing(m, tok, TEXTS)
    assert len(rows) == len(all_components(m))
    base = health(m, tok, TEXTS)
    assert all((r.health.accuracy < 0.5 * base.accuracy) == r.broken for r in rows)


def test_dead_features_are_found():
    m, tok = tiny_model(), FakeTok()
    with torch.no_grad():  # kill neuron 5 of layer 1: zero gate and up rows -> activation exactly 0
        m.model.layers[1].mlp.gate_proj.weight[5] = 0
        m.model.layers[1].mlp.up_proj.weight[5] = 0
    dead = dead_features(m, tok, TEXTS)
    assert 5 in dead[1] and len(dead[0]) == 0


def test_null_model_keeps_values_but_not_structure():
    m = tiny_model()
    n = null_model(m, seed=1)
    a, b = m.model.layers[0].mlp.gate_proj.weight, n.model.layers[0].mlp.gate_proj.weight
    assert not torch.equal(a, b)
    assert torch.equal(a.flatten().sort().values, b.flatten().sort().values)


def test_study_edit_reports_health():
    m, tok = tiny_model(), FakeTok()
    battery = [Probe("the capital of France is", "Paris", ("target",)),
               Probe("the capital of Italy is", "Rome", ("control",))]
    rep = study_edit(m, tok, suppress(m, [(1, 3), (2, 7)]), battery, health_texts=TEXTS)
    hb, ha = rep.health
    assert 0 <= hb.accuracy <= 1 and 0 <= ha.accuracy <= 1
    assert "health" in rep.summary()


def test_dead_features_finds_near_silent_neurons():
    # SiLU is almost never exactly 0: a neuron scaled ~1000x below its layer
    # is dead in practice, yet still above an absolute 1e-6
    m, tok = tiny_model(), FakeTok()
    with torch.no_grad():
        m.model.layers[1].mlp.gate_proj.weight[9] *= 3e-2
        m.model.layers[1].mlp.up_proj.weight[9] *= 3e-2
    assert 9 in dead_features(m, tok, TEXTS)[1]
    assert 9 not in dead_features(m, tok, TEXTS, tol=1e-6)[1]


def test_compare_scale_flags_a_mismatched_model():
    import copy

    from marv.diagnostics import compare_scale

    m, tok = tiny_model(), FakeTok()
    same = compare_scale(m, m, tok, "the capital of France is")
    assert same.ok and abs(same.final_ratio - 1) < 1e-6
    big = copy.deepcopy(m)
    with torch.no_grad():
        for blk in big.model.layers:
            blk.mlp.down_proj.weight *= 1000
            blk.self_attn.o_proj.weight *= 1000
    assert not compare_scale(m, big, tok, "the capital of France is").ok


def test_mean_writes_skips_the_first_position():
    m, tok = tiny_model(), FakeTok()
    text = "the capital of France is"
    w = capture_writes(m, tok, text, positions=list(range(5)))
    got = mean_writes(m, tok, [text], [(1, "mlp")])[(1, "mlp")]
    torch.testing.assert_close(got, w.parts[(1, "mlp")][1:].mean(0))
    got_all = mean_writes(m, tok, [text], [(1, "mlp")], skip_first=False)[(1, "mlp")]
    torch.testing.assert_close(got_all, w.parts[(1, "mlp")].mean(0))
