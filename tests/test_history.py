"""History on a tiny synthetic Llama, no network. A regression is planted in
four FFN neurons at step 9 of 16 fake checkpoints, with small harmless drift
in other neurons at every step: bisect must find step 9, blame must name the
four and prove it by reverting them, gate must fail the bad version and pass
a good one."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from marv.context import hidden_states_at_layers
from marv.evaluate import Probe
from marv.history import BatteryTest, History, changed_neurons, mcnemar_regression_p, revert_neurons
from test_edit import FakeTok, tiny_model

WORDS = ["the", "capital", "of", "France", "is", "Paris", "Germany", "Berlin", "Italy", "Rome",
         "language", "French", "German", "weather", "in", "Tokyo", "Japan", "a", "and", "to"]
PLANT_LAYER, PLANTED, PLANT_STEP = 2, [3, 11, 17, 25], 9


@pytest.fixture(scope="module")
def world():
    tok = FakeTok()
    rng = np.random.default_rng(0)
    prompts = sorted({" ".join(rng.choice(WORDS, size=4)) for _ in range(60)})[:24]
    base = tiny_model().eval()
    with torch.no_grad():
        targets = [int(base(**tok(p)).logits[0, -1].argmax()) for p in prompts]
    # the base model is right on every item by construction
    test = BatteryTest("facts", [Probe(p, "x", ("t",), target_ids=(t,)) for p, t in zip(prompts, targets)])
    x = np.stack([hidden_states_at_layers(base, tok, p, [PLANT_LAYER])[PLANT_LAYER] for p in prompts]).mean(0)
    direction = torch.tensor(x / np.linalg.norm(x), dtype=torch.float32)

    def make(step: int):
        m = copy.deepcopy(base)
        g = torch.Generator().manual_seed(100)
        with torch.no_grad():
            for s in range(1, step + 1):  # harmless drift: tiny nudges to other neurons, every step
                f = int(torch.randint(0, 32, (1,), generator=g))  # layer 1; the plant is in layer 2
                m.model.layers[1].mlp.down_proj.weight[:, f] += 1e-4 * torch.randn(16, generator=g)
            if step >= PLANT_STEP:
                mlp = m.model.layers[PLANT_LAYER].mlp
                wu = m.lm_head.weight[40]
                for f in PLANTED:
                    mlp.gate_proj.weight[f] = 6 * direction
                    mlp.up_proj.weight[f] = 6 * direction
                    mlp.down_proj.weight[:, f] = 3 * wu / wu.norm()
        return m.eval()

    labels = [f"step-{s}" for s in range(16)]
    return tok, test, make, labels


def load_from(make):
    return lambda label: make(int(label.split("-")[1]))


def test_commit_log_and_never_rerun(tmp_path, world):
    tok, test, make, _ = world
    h = History(str(tmp_path / "h.sqlite"))
    h.commit(make(0), tok, "step-0", [test], step=0)
    h.commit(make(PLANT_STEP), tok, "step-9", [test], step=9)
    log = h.log("facts")
    assert [r.label for r in log.rows] == ["step-0", "step-9"]
    assert log.rows[0].accuracy == 1.0 and log.rows[1].accuracy < 0.5
    runs = h.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    h.commit(make(0), tok, "step-0", [test])  # same weights, same test: nothing reruns
    assert h.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == runs
    with pytest.raises(ValueError, match="different weights"):
        h.commit(make(PLANT_STEP), tok, "step-0", [test])
    with pytest.raises(ValueError, match="different items"):
        h.commit(make(0), tok, "step-0", [BatteryTest("facts", test.probes[:3])])


def test_jsonl_roundtrip_is_identical(tmp_path, world):
    tok, test, make, _ = world
    h = History(str(tmp_path / "a.sqlite"))
    h.commit(make(0), tok, "step-0", [test], step=0)
    h.commit(make(12), tok, "step-12", [test], step=12)
    h.export_jsonl(str(tmp_path / "r.jsonl"))
    h2 = History.from_jsonl(str(tmp_path / "r.jsonl"), str(tmp_path / "b.sqlite"))
    for table in ("versions", "tests", "runs", "results"):
        q = f"SELECT * FROM {table} ORDER BY rowid"
        assert h.db.execute(q).fetchall() == h2.db.execute(q).fetchall()


def test_failed_check_is_never_reported(tmp_path, world):
    tok, test, make, _ = world
    h = History(str(tmp_path / "h.sqlite"))
    h.commit(make(0), tok, "step-0", [test])
    h.db.execute("UPDATE runs SET check_ok=0")
    assert h.log("facts").rows == [] and h.log("facts").hidden_failed_checks == 1
    with pytest.raises(ValueError, match="failed its check"):
        h.results("step-0", "facts")


def test_bisect_finds_the_planted_step(tmp_path, world):
    tok, test, make, labels = world
    h = History(str(tmp_path / "h.sqlite"))
    r = h.bisect(test, labels, load_from(make), tok)
    assert r.first_changed == f"step-{PLANT_STEP}" and r.last_unchanged == f"step-{PLANT_STEP - 1}"
    assert r.runs <= 6  # 16 versions: first + last + ~log2(16)
    again = h.bisect(test, labels, load_from(make), tok)
    assert again.first_changed == r.first_changed and again.runs == 0  # stored results reused


def test_blame_names_the_planted_neurons_and_proves_it(tmp_path, world):
    tok, test, make, _ = world
    h = History(str(tmp_path / "h.sqlite"))
    before, after = "step-8", "step-9"
    h.commit(make(8), tok, before, [test])
    h.commit(make(9), tok, after, [test])
    suspects = changed_neurons(make(8), make(9))
    found = {n for n, _ in suspects}
    assert {(PLANT_LAYER, f) for f in PLANTED} <= found  # the plant, plus step 9's own drift nudge
    assert len(found - {(PLANT_LAYER, f) for f in PLANTED}) == 1

    far = h.blame(test, "step-0", "step-15", load_from(make), tok)  # many drifted suspects too
    assert len(far.suspects) > len(PLANTED)
    assert far.ceiling > 0.9
    assert set(far.culprits) <= {(PLANT_LAYER, f) for f in PLANTED}
    assert far.restored >= 0.9
    assert far.runs < len(far.suspects)  # halving beats reverting one at a time
    far.show()


def test_revert_neurons_restores_exactly(world):
    tok, _, make, _ = world
    good, bad = make(0), make(PLANT_STEP)
    snapshot = {k: v.clone() for k, v in bad.state_dict().items()}
    with revert_neurons(bad, good, [(PLANT_LAYER, f) for f in PLANTED]):
        for f in PLANTED:
            assert torch.equal(bad.model.layers[PLANT_LAYER].mlp.down_proj.weight[:, f],
                               good.model.layers[PLANT_LAYER].mlp.down_proj.weight[:, f])
    assert all(torch.equal(v, snapshot[k]) for k, v in bad.state_dict().items())


def test_gate_fails_the_bad_version_and_passes_a_good_one(tmp_path, world):
    tok, test, make, _ = world
    h = History(str(tmp_path / "h.sqlite"))
    for s in (0, 5, PLANT_STEP):
        h.commit(make(s), tok, f"step-{s}", [test])
    assert h.gate("step-5", "step-0", ["facts"]).passed  # drift only
    rep = h.gate(f"step-{PLANT_STEP}", "step-0", ["facts"])
    assert not rep.passed and rep.tests[0].right_to_wrong > 5
    rep.show()


def test_mcnemar_and_power():
    assert mcnemar_regression_p(0, 0) == 1.0
    assert mcnemar_regression_p(5, 0) == pytest.approx(1 / 32)
    assert mcnemar_regression_p(3, 3) > 0.5
