"""History: test model versions, keep every result, find what changed and why.

Design: docs/history-design.md. One SQLite file holds versions, tests and
per-question results; weights are never stored here (Hugging Face versions
them). On top of that record:

    h = History()                         # creates ./.marv/history.sqlite
    h.commit(model, tok, "base", [capitals])
    h.commit(tuned, tok, "tuned", [capitals])
    h.log("capitals").show()              # one test across versions
    h.gate("tuned", "base", ["capitals"]).show()      # paired: did it get worse?
    h.bisect(capitals, labels, load, tok)              # where did it change?
    h.blame(capitals, "base", "tuned", load, tok)      # which neurons? proved by reverting them

Rules carried over from MARV: a run whose check failed is stored but never
reported; a weight diff is only a suspect list, and only the revert test
(a TOTAL effect) names a cause.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import torch

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY, label TEXT UNIQUE NOT NULL, model_id TEXT, revision TEXT,
    step INTEGER, parent TEXT, dtype TEXT, exact INTEGER, weights_sha256 TEXT,
    created_at TEXT, marv_version TEXT, torch_version TEXT, transformers_version TEXT);
CREATE TABLE IF NOT EXISTS source_files (
    version_id INTEGER, filename TEXT, sha256 TEXT, PRIMARY KEY (version_id, filename));
CREATE TABLE IF NOT EXISTS tests (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, kind TEXT, spec_json TEXT, spec_hash TEXT);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, version_id INTEGER, test_id INTEGER, env_json TEXT,
    check_ok INTEGER, started_at TEXT, UNIQUE (version_id, test_id));
CREATE TABLE IF NOT EXISTS results (
    run_id INTEGER, item TEXT, correct INTEGER, score REAL, PRIMARY KEY (run_id, item));
"""
TABLES = ("versions", "source_files", "tests", "runs", "results")


# ------------------------------------------------------------------ tests
class Test:
    """A named set of items, each scored right/wrong plus a number. Subclass
    and implement `spec` (JSON-able, fixes the items) and `run`."""

    kind = "custom"

    def __init__(self, name: str):
        self.name = name

    def spec(self) -> dict:
        raise NotImplementedError

    def run(self, model, tokenizer, device: str = "cpu") -> tuple[list[tuple[str, bool, float]], bool]:
        """Returns ([(item_id, correct, score), ...], check_ok)."""
        raise NotImplementedError


class BatteryTest(Test):
    """A MARV probe battery as a test: an item is correct when its target is
    the top-1 prediction; its score is the target's probability."""

    kind = "battery"

    def __init__(self, name: str, probes):
        super().__init__(name)
        self.probes = list(probes)
        prompts = [p.prompt for p in self.probes]
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"test {name!r}: prompts must be unique (they are the item ids)")

    def spec(self) -> dict:
        return {"probes": [{"prompt": p.prompt, "target": p.target, "tags": list(p.tags),
                            "target_ids": None if p.target_ids is None else [int(t) for t in p.target_ids]}
                           for p in self.probes]}

    def run(self, model, tokenizer, device: str = "cpu"):
        from .evaluate import run_battery

        rows = run_battery(model, tokenizer, self.probes, device).rows
        items = [(r.prompt, r.target_rank == 1, float(r.target_prob)) for r in rows]
        ok = all(math.isfinite(s) and 0.0 <= s <= 1.0 + 1e-6 for _, _, s in items)
        return items, ok


def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ provenance
def weights_sha256(model) -> str:
    """SHA-256 over every parameter's name, dtype, shape and bytes: the exact
    identity of a set of weights, whatever its revision label says."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        t = p.detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def code_versions() -> dict:
    """MARV (version plus git commit when run from a checkout), torch,
    transformers: which code produced a number."""
    from ._version import __version__

    marv = __version__
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(repo, ".git")):
        try:
            sha = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True).stdout.strip()
            marv += f"+{sha}" + (".dirty" if dirty else "")
        except (OSError, subprocess.CalledProcessError):
            pass
    import transformers

    return {"marv": marv, "torch": torch.__version__, "transformers": transformers.__version__}


def hf_source_checksums(model_id: str, revision: str | None) -> dict[str, str]:
    """SHA-256 of each weight file in a Hugging Face repo at `revision`, as the
    Hub reports them. Empty if offline or not a Hub model."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    except Exception:
        return {}
    return {s.rfilename: s.lfs.sha256 for s in info.siblings or []
            if getattr(s, "lfs", None) is not None and s.rfilename.endswith((".safetensors", ".bin"))}


# ------------------------------------------------------------------ statistics
def mcnemar_regression_p(right_to_wrong: int, wrong_to_right: int) -> float:
    """One-sided exact McNemar: P(at least this many right->wrong flips) if
    flips were equally likely in both directions."""
    n = right_to_wrong + wrong_to_right
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(right_to_wrong, n + 1)) / 2 ** n)


def min_detectable_flips(alpha: float) -> int:
    """Fewest right->wrong flips (with none the other way) that can reach p < alpha."""
    return math.ceil(math.log(1 / alpha, 2))


def paired_bootstrap_ci(diffs: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(reps, len(diffs)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ reports
@dataclass
class LogRow:
    label: str
    step: int | None
    created_at: str
    n: int
    accuracy: float
    mean_score: float


@dataclass
class Log:
    test: str
    rows: list[LogRow]
    hidden_failed_checks: int = 0

    def show(self) -> str:
        lines = [f"log: {self.test}", f"  {'version':<20} {'step':>6} {'n':>5} {'accuracy':>9} {'mean score':>11}"]
        for r in self.rows:
            step = "" if r.step is None else str(r.step)
            lines.append(f"  {r.label:<20} {step:>6} {r.n:>5} {r.accuracy:>9.3f} {r.mean_score:>11.4f}")
        if self.hidden_failed_checks:
            lines.append(f"  ({self.hidden_failed_checks} run(s) hidden: their check failed)")
        out = "\n".join(lines)
        print(out)
        return out


@dataclass
class TestComparison:
    test: str
    n: int  # items scored in both versions
    right_to_wrong: int
    wrong_to_right: int
    p_regression: float
    mean_score_change: float
    score_change_ci: tuple[float, float]
    regressed: bool
    underpowered: bool  # too few items for any regression to reach significance


@dataclass
class GateReport:
    candidate: str
    current: str
    alpha: float
    tests: list[TestComparison]

    @property
    def passed(self) -> bool:
        return not any(t.regressed for t in self.tests)

    def show(self) -> str:
        lines = [f"gate: {self.candidate} vs {self.current}  ->  {'PASS' if self.passed else 'FAIL'}"]
        for t in self.tests:
            flag = "REGRESSED" if t.regressed else "ok"
            lines.append(f"  {t.test:<20} n={t.n:<4} right->wrong {t.right_to_wrong:<3} wrong->right {t.wrong_to_right:<3} "
                         f"p={t.p_regression:.3g}  score {t.mean_score_change:+.4f} "
                         f"[{t.score_change_ci[0]:+.4f}, {t.score_change_ci[1]:+.4f}]  {flag}")
            if t.underpowered:
                lines.append(f"    warning: {t.n} items cannot show a regression at alpha={self.alpha} "
                             f"(needs at least {min_detectable_flips(self.alpha)} flips)")
        out = "\n".join(lines)
        print(out)
        return out


@dataclass
class BisectResult:
    first_changed: str | None
    last_unchanged: str | None
    tested: list[tuple[str, bool]]  # (version, changed?) in the order they were checked
    runs: int  # versions actually run (stored results are reused, not rerun)


@dataclass
class BlameResult:
    """TOTAL effects from revert tests: the after-model with some neurons'
    parameters copied back from the before-model, rerun on the items that
    regressed. `restored` is the share of the lost score that comes back."""

    items: list[str]  # the items that went right -> wrong
    suspects: list[tuple[tuple[int, int], float]]  # ((layer, neuron), relative change), most changed first
    ceiling: float  # restored when every changed MLP neuron is reverted
    culprits: list[tuple[int, int]]  # smallest set found that restores >= threshold
    restored: float  # restored by `culprits` alone
    runs: int
    note: str = ""

    def show(self) -> str:
        lines = [f"blame: {len(self.items)} regressed item(s), {len(self.suspects)} changed MLP neurons (suspects)",
                 f"  revert all suspects: restores {self.ceiling:.0%}  [total effect]"]
        if self.culprits:
            names = ", ".join(f"L{l} n{f}" for l, f in self.culprits[:12])
            more = f" (+{len(self.culprits) - 12})" if len(self.culprits) > 12 else ""
            lines.append(f"  culprits: {names}{more}  restore {self.restored:.0%} on their own")
        if self.note:
            lines.append(f"  note: {self.note}")
        lines.append(f"  model runs: {self.runs}")
        out = "\n".join(lines)
        print(out)
        return out


# ------------------------------------------------------------------ revert
@contextmanager
def revert_neurons(model, base, neurons):
    """Within the block, `model`'s FFN neurons [(layer, neuron), ...] carry
    `base`'s parameters (gate row, up row, down column; every slice the
    adapter lists). Restored on exit."""
    from .arch import detect_adapter

    ad_m, ad_b = detect_adapter(model), detect_adapter(base)
    by_layer: dict[int, list[int]] = {}
    for layer, f in neurons:
        by_layer.setdefault(int(layer), []).append(int(f))
    saved = []
    try:
        with torch.no_grad():
            for layer, fs in by_layer.items():
                idx = torch.tensor(fs)
                for (pm, axis), (pb, _) in zip(ad_m.neuron_params(model, layer), ad_b.neuron_params(base, layer)):
                    saved.append((pm, axis, idx, pm.index_select(axis, idx.to(pm.device)).clone()))
                    pm.index_copy_(axis, idx.to(pm.device), pb.index_select(axis, idx.to(pb.device)).to(pm.device, pm.dtype))
        yield
    finally:
        with torch.no_grad():
            for pm, axis, idx, old in saved:
                pm.index_copy_(axis, idx.to(pm.device), old)


def changed_neurons(before, after) -> list[tuple[tuple[int, int], float]]:
    """Every FFN neuron whose parameters differ at all, with its relative
    change (||delta|| / ||before|| over all its slices), most changed first.
    A suspect list, not an explanation."""
    from .arch import detect_adapter

    ad_b, ad_a = detect_adapter(before), detect_adapter(after)
    out = []
    for layer in range(ad_b.num_layers(before)):
        sq_delta = sq_base = None
        for (pb, axis), (pa, _) in zip(ad_b.neuron_params(before, layer), ad_a.neuron_params(after, layer)):
            b, a = pb.detach().float().cpu(), pa.detach().float().cpu()
            if b.shape != a.shape:
                raise ValueError(f"layer {layer}: parameter shapes differ ({tuple(b.shape)} vs {tuple(a.shape)})")
            other = tuple(d for d in range(b.dim()) if d != axis)
            d2 = ((a - b) ** 2).sum(dim=other)
            b2 = (b ** 2).sum(dim=other)
            sq_delta = d2 if sq_delta is None else sq_delta + d2
            sq_base = b2 if sq_base is None else sq_base + b2
        rel = torch.sqrt(sq_delta / sq_base.clamp_min(1e-30))
        for f in torch.nonzero(sq_delta > 0).flatten().tolist():
            out.append(((layer, f), float(rel[f])))
    out.sort(key=lambda kv: -kv[1])
    return out


# ------------------------------------------------------------------ History
class History:
    """The results database. `path` defaults to ./.marv/history.sqlite
    (/content/.marv/ on Colab), created on first use."""

    def __init__(self, path: str | None = None):
        self.path = os.path.abspath(path or os.path.join(".marv", "history.sqlite"))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---- versions and tests
    def _version(self, label: str):
        return self.db.execute("SELECT * FROM versions WHERE label=?", (label,)).fetchone()

    def versions(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT label FROM versions ORDER BY id")]

    def _test_id(self, test: Test) -> int:
        spec = test.spec()
        h = _spec_hash(spec)
        row = self.db.execute("SELECT id, spec_hash FROM tests WHERE name=?", (test.name,)).fetchone()
        if row is not None:
            if row[1] != h:
                raise ValueError(f"test {test.name!r} already exists with different items; give the new one another name")
            return row[0]
        cur = self.db.execute("INSERT INTO tests (name, kind, spec_json, spec_hash) VALUES (?,?,?,?)",
                              (test.name, test.kind, json.dumps(spec, sort_keys=True), h))
        return cur.lastrowid

    def has_run(self, label: str, test_name: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM runs r JOIN versions v ON v.id=r.version_id JOIN tests t ON t.id=r.test_id "
            "WHERE v.label=? AND t.name=?", (label, test_name)).fetchone() is not None

    def commit(self, model, tokenizer, label: str, tests=(), *, model_id: str | None = None,
               revision: str | None = None, step: int | None = None, parent: str | None = None,
               fetch_checksums: bool = True, device: str = "cpu") -> str:
        """Record `model` as version `label` and run every test in `tests` it
        has not run yet. Committing the same label again only runs new tests;
        a label that already names different weights is refused."""
        cfg = getattr(model, "config", None)
        model_id = model_id or getattr(cfg, "_name_or_path", "") or None
        revision = revision or getattr(cfg, "_commit_hash", None)
        digest = weights_sha256(model)
        row = self._version(label)
        if row is not None:
            if row[8] != digest:
                raise ValueError(f"version {label!r} already exists with different weights; use a new label")
            version_id = row[0]
        else:
            params = list(model.parameters())
            code = code_versions()
            cur = self.db.execute(
                "INSERT INTO versions (label, model_id, revision, step, parent, dtype, exact, weights_sha256, "
                "created_at, marv_version, torch_version, transformers_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (label, model_id, revision, step, parent, str(params[0].dtype).replace("torch.", ""),
                 int(getattr(cfg, "quantization_config", None) is None), digest,
                 datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 code["marv"], code["torch"], code["transformers"]))
            version_id = cur.lastrowid
            if fetch_checksums and model_id and revision:
                for fn, sha in hf_source_checksums(model_id, revision).items():
                    self.db.execute("INSERT INTO source_files VALUES (?,?,?)", (version_id, fn, sha))
        for test in tests:
            test_id = self._test_id(test)
            if self.db.execute("SELECT 1 FROM runs WHERE version_id=? AND test_id=?",
                               (version_id, test_id)).fetchone():
                continue  # never rerun: the result is already recorded
            items, ok = test.run(model, tokenizer, device)
            env = {**code_versions(), "device": device}
            cur = self.db.execute(
                "INSERT INTO runs (version_id, test_id, env_json, check_ok, started_at) VALUES (?,?,?,?,?)",
                (version_id, test_id, json.dumps(env, sort_keys=True), int(ok),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")))
            self.db.executemany("INSERT INTO results VALUES (?,?,?,?)",
                                [(cur.lastrowid, item, int(c), float(s)) for item, c, s in items])
        self.db.commit()
        return label

    def results(self, label: str, test_name: str) -> dict[str, tuple[bool, float]]:
        """{item: (correct, score)} for one version and test. Raises if the run
        is missing or its check failed: those numbers are never reported."""
        row = self.db.execute(
            "SELECT r.id, r.check_ok FROM runs r JOIN versions v ON v.id=r.version_id "
            "JOIN tests t ON t.id=r.test_id WHERE v.label=? AND t.name=?", (label, test_name)).fetchone()
        if row is None:
            raise KeyError(f"no run of {test_name!r} on {label!r}")
        if not row[1]:
            raise ValueError(f"the run of {test_name!r} on {label!r} failed its check; its numbers are not reported")
        return {item: (bool(c), s) for item, c, s in
                self.db.execute("SELECT item, correct, score FROM results WHERE run_id=?", (row[0],))}

    # ---- log
    def log(self, test_name: str) -> Log:
        rows, hidden = [], 0
        for label, step, created, run_id, ok in self.db.execute(
                "SELECT v.label, v.step, v.created_at, r.id, r.check_ok FROM runs r "
                "JOIN versions v ON v.id=r.version_id JOIN tests t ON t.id=r.test_id "
                "WHERE t.name=? ORDER BY v.id", (test_name,)).fetchall():
            if not ok:
                hidden += 1
                continue
            res = self.db.execute("SELECT correct, score FROM results WHERE run_id=?", (run_id,)).fetchall()
            n = len(res)
            rows.append(LogRow(label, step, created, n,
                               sum(c for c, _ in res) / n if n else float("nan"),
                               sum(s for _, s in res) / n if n else float("nan")))
        return Log(test_name, rows, hidden)

    # ---- gate
    def compare(self, candidate: str, current: str, test_name: str, alpha: float = 0.05) -> TestComparison:
        a, b = self.results(current, test_name), self.results(candidate, test_name)
        common = sorted(set(a) & set(b))
        r2w = sum(a[i][0] and not b[i][0] for i in common)
        w2r = sum(b[i][0] and not a[i][0] for i in common)
        diffs = np.array([b[i][1] - a[i][1] for i in common])
        p = mcnemar_regression_p(r2w, w2r)
        return TestComparison(test_name, len(common), r2w, w2r, p,
                              float(diffs.mean()) if len(diffs) else float("nan"),
                              paired_bootstrap_ci(diffs), regressed=(r2w > w2r and p < alpha),
                              underpowered=len(common) < min_detectable_flips(alpha))

    def gate(self, candidate: str, current: str, test_names, alpha: float = 0.05) -> GateReport:
        """Same tests, same items, both versions: FAIL if any test shows a
        significant right->wrong regression (one-sided exact McNemar). Both
        versions must already be committed with these tests."""
        return GateReport(candidate, current, alpha, [self.compare(candidate, current, t, alpha) for t in test_names])

    # ---- bisect
    def bisect(self, test: Test, labels: list[str], load, tokenizer, *, item: str | None = None,
               alpha: float = 0.05, device: str = "cpu") -> BisectResult:
        """Find the first version in the ordered `labels` whose result on `test`
        differs from the first one's. `load(label)` returns that version's
        model; versions already committed with this test are not rerun.
        "Differs" means: `item` flipped (if given), else a significant
        right->wrong regression. Assumes the change persists once it appears."""
        good, runs, tested = labels[0], 0, []

        def ensure(label):
            nonlocal runs
            if not self.has_run(label, test.name):
                model = load(label)
                self.commit(model, tokenizer, label, [test], fetch_checksums=False, device=device)
                runs += 1
                del model

        def changed(label):
            ensure(label)
            if item is not None:
                return self.results(label, test.name)[item][0] != self.results(good, test.name)[item][0]
            return self.compare(label, good, test.name, alpha).regressed

        ensure(good)
        if not changed(labels[-1]):
            tested.append((labels[-1], False))
            return BisectResult(None, labels[-1], tested, runs)
        tested.append((labels[-1], True))
        lo, hi = 0, len(labels) - 1  # labels[lo] unchanged, labels[hi] changed
        while hi - lo > 1:
            mid = (lo + hi) // 2
            c = changed(labels[mid])
            tested.append((labels[mid], c))
            lo, hi = (lo, mid) if c else (mid, hi)
        return BisectResult(labels[hi], labels[lo], tested, runs)

    # ---- blame
    def blame(self, test: Test, before: str, after: str, load, tokenizer, *, threshold: float = 0.9,
              device: str = "cpu") -> BlameResult:
        """Which changed FFN neurons caused the items that went right -> wrong
        between `before` and `after`? Diff the weights (suspects), then revert:
        all suspects first (the ceiling), then halve the suspect list while
        what remains still restores >= `threshold` of the lost score. Both
        versions are committed (with `test`) if they are not already."""
        if not before or not after:
            raise ValueError(f"blame needs two version labels, got {before!r} and {after!r}")
        m_before, m_after = load(before), load(after)
        for label, model in ((before, m_before), (after, m_after)):
            self.commit(model, tokenizer, label, [test], fetch_checksums=False, device=device)
        a, b = self.results(before, test.name), self.results(after, test.name)
        items = sorted(i for i in set(a) & set(b) if a[i][0] and not b[i][0])
        if not items:
            return BlameResult([], [], 0.0, [], 0.0, 0, "no item went right -> wrong")
        suspects = changed_neurons(m_before, m_after)
        sub = [p for p in getattr(test, "probes", []) if p.prompt in items]
        if not sub:
            raise TypeError("blame needs a BatteryTest (it reruns the regressed probes)")
        sub_test = BatteryTest(test.name + ":regressed", sub)
        lost = sum(a[i][1] for i in items) - sum(b[i][1] for i in items)
        runs = 0

        def restored(neurons) -> float:
            nonlocal runs
            runs += 1
            with revert_neurons(m_after, m_before, neurons):
                got, _ = sub_test.run(m_after, tokenizer, device)
            gain = sum(s for _, _, s in got) - sum(b[i][1] for i in items)
            return gain / lost if lost > 0 else float("nan")

        ceiling = restored([n for n, _ in suspects]) if suspects else 0.0
        if not suspects or ceiling < threshold:
            note = ("no FFN neuron changed: the change is elsewhere (attention, norms, embeddings)" if not suspects else
                    f"reverting every changed FFN neuron restores only {ceiling:.0%}: part of the change is "
                    "outside the FFN neurons (attention, norms, embeddings), or the neurons act with them")
            return BlameResult(items, suspects, ceiling, [], 0.0, runs, note)
        group, score, note = [n for n, _ in suspects], ceiling, ""
        while len(group) > 1:
            half = len(group) // 2
            first, second = group[:half], group[half:]
            s1 = restored(first)
            if s1 >= threshold:
                group, score = first, s1
                continue
            s2 = restored(second)
            if s2 >= threshold:
                group, score = second, s2
                continue
            note = (f"stopped at {len(group)} neurons: neither half restores {threshold:.0%} on its own "
                    f"({s1:.0%}, {s2:.0%}); they matter jointly")
            break
        return BlameResult(items, suspects, ceiling, group, score, runs, note)

    # ---- moving the file
    def export_jsonl(self, path: str) -> str:
        """Every row of every table as JSON lines: plain text you can commit to
        GitHub, diff and merge. `History.from_jsonl` rebuilds the database."""
        with open(path, "w") as f:
            for table in TABLES:
                cols = [c[1] for c in self.db.execute(f"PRAGMA table_info({table})")]
                for row in self.db.execute(f"SELECT * FROM {table} ORDER BY rowid"):
                    f.write(json.dumps({"table": table, "row": dict(zip(cols, row))}, sort_keys=True) + "\n")
        return path

    @classmethod
    def from_jsonl(cls, jsonl_path: str, path: str | None = None) -> "History":
        h = cls(path)
        if h.versions():
            raise ValueError(f"{h.path} is not empty; rebuild into a new path")
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["table"] not in TABLES:
                    raise ValueError(f"unknown table {rec['table']!r}")
                cols = sorted(rec["row"])
                h.db.execute(f"INSERT INTO {rec['table']} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                             [rec["row"][c] for c in cols])
        h.db.commit()
        return h

    def download(self) -> str:
        """On Colab, download history.sqlite to your computer (Colab's
        /content is wiped when the session ends). Elsewhere, return its path."""
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if "google.colab" in sys.modules:
            from google.colab import files

            files.download(self.path)
        return self.path

    @classmethod
    def upload(cls, path: str | None = None) -> "History":
        """On Colab, pick a history.sqlite or results.jsonl from your computer
        to continue from."""
        from google.colab import files

        name, data = next(iter(files.upload().items()))
        if name.endswith(".jsonl"):
            tmp = os.path.join("/tmp", name)
            open(tmp, "wb").write(data)
            return cls.from_jsonl(tmp, path)
        target = os.path.abspath(path or os.path.join(".marv", "history.sqlite"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            raise FileExistsError(f"{target} exists; pass another path")
        open(target, "wb").write(data)
        return cls(target)

    def close(self):
        self.db.close()


# ------------------------------------------------------------------ training
def history_callback(history: History, tokenizer, tests, label: str = "step-{step}",
                     model_id: str | None = None):
    """A transformers TrainerCallback that commits every saved checkpoint to
    `history` and runs `tests` on it, so a training run leaves a full record:

        trainer = Trainer(..., callbacks=[marv.history_callback(h, tok, [facts, controls])])

    Keep the checkpoints (no `save_total_limit`, or `push_to_hub=True` with
    `hub_strategy="every_save"`) if you want to bisect or blame later: History
    stores results, not weights."""
    from transformers import TrainerCallback

    class _HistoryCallback(TrainerCallback):
        def __init__(self):
            self.parent = None

        def on_save(self, args, state, control, model=None, **kwargs):
            m = getattr(model, "module", model)
            was_training = m.training
            m.eval()
            name = label.format(step=state.global_step)
            device = str(next(m.parameters()).device)
            history.commit(m, tokenizer, name, tests, model_id=model_id, step=state.global_step,
                           parent=self.parent, fetch_checksums=False, device=device)
            m.train(was_training)
            self.parent = name

    return _HistoryCallback()


def checkpoint_loader(output_dir: str, model_class=None, **from_pretrained_kwargs):
    """`load(label)` for bisect/blame over a Trainer's local checkpoints: the
    trailing number of a label ("step-40") picks `output_dir/checkpoint-40`."""
    import re

    if model_class is None:
        from transformers import AutoModelForCausalLM as model_class

    def load(label: str):
        m = re.search(r"(\d+)$", label)
        if m is None:
            raise ValueError(f"label {label!r} does not end in a step number")
        path = os.path.join(output_dir, f"checkpoint-{m.group(1)}")
        return model_class.from_pretrained(path, **from_pretrained_kwargs).eval()

    return load
