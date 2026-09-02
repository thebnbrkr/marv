"""Probe batteries: run a fixed set of prompts through a model, before and
after an edit, and report exactly what moved.

This is the collateral-damage microscope. Point it at (base, finetune),
(fp16, int4), or (model, model-under-an-edit) and get a per-prompt table:
which predictions flipped, which degraded, which held. `tags` group the
prompts so "broke 3 capitals, held 40 unrelated" is one line.

Small models are the point: on a 135M model you can afford a few hundred
probes per edit and actually chart the efficacy/specificity frontier that
knowledge-editing papers on 7B models can only sample.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch


@dataclass
class Probe:
    prompt: str
    target: str  # expected continuation; its first token is the one scored
    tags: tuple[str, ...] = ()


@dataclass
class ProbeRow:
    prompt: str
    target: str
    target_id: int
    top1: str
    top1_id: int
    target_rank: int
    target_prob: float
    tags: tuple[str, ...]


@dataclass
class BatteryResult:
    rows: list[ProbeRow]

    def by_prompt(self) -> dict[str, ProbeRow]:
        return {r.prompt: r for r in self.rows}


def _first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(" " + text.strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"target {text!r} tokenized to nothing")
    return int(ids[0])


@torch.no_grad()
def run_battery(model, tokenizer, probes, device: str = "cpu") -> BatteryResult:
    """Forward each probe once; record the top-1 next token and the target
    token's rank + probability."""
    model.eval()
    rows: list[ProbeRow] = []
    for p in probes:
        enc = tokenizer(p.prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        tid = _first_token_id(tokenizer, p.target)
        order = torch.argsort(logits, descending=True)
        rank = int((order == tid).nonzero()[0, 0]) + 1
        top1_id = int(order[0])
        rows.append(
            ProbeRow(
                prompt=p.prompt,
                target=p.target,
                target_id=tid,
                top1=tokenizer.decode([top1_id]).strip(),
                top1_id=top1_id,
                target_rank=rank,
                target_prob=float(probs[tid]),
                tags=tuple(p.tags),
            )
        )
    return BatteryResult(rows)


@dataclass
class DiffRow:
    prompt: str
    tags: tuple[str, ...]
    top1_before: str
    top1_after: str
    prob_before: float
    prob_after: float
    rank_before: int
    rank_after: int
    verdict: str  # "flipped" | "degraded" | "improved" | "unchanged"

    @property
    def impact(self) -> float:
        return abs(self.prob_after - self.prob_before)


_MARKS = {"flipped": "x", "degraded": "x", "improved": "+", "unchanged": "."}
_ORDER = {"flipped": 0, "degraded": 1, "improved": 2, "unchanged": 3}


def _verdict(b: ProbeRow, a: ProbeRow, eps: float = 0.05) -> str:
    if b.top1_id != a.top1_id:
        return "flipped"
    dp = a.target_prob - b.target_prob
    if dp < -eps:
        return "degraded"
    if dp > eps:
        return "improved"
    return "unchanged"


@dataclass
class BatteryDiff:
    rows: list[DiffRow] = field(default_factory=list)

    def changed(self) -> list[DiffRow]:
        return [r for r in self.rows if r.verdict != "unchanged"]

    def by_tag(self) -> dict[str, Counter]:
        out: dict[str, Counter] = {}
        for r in self.rows:
            for t in r.tags:
                out.setdefault(t, Counter())[r.verdict] += 1
        return out

    def metrics(self) -> dict[str, dict[str, float]]:
        """Per-tag summary: `moved` fraction (flipped or degraded or improved),
        `mean_dprob` (signed mean target-prob change), `n`. Read it as the
        editing literature's efficacy / specificity axes -- high `moved` on
        your target tag is efficacy, high `moved` on neighbour/control tags
        is collateral."""
        groups: dict[str, list[DiffRow]] = {}
        for r in self.rows:
            for t in r.tags:
                groups.setdefault(t, []).append(r)
        groups["_all"] = list(self.rows)
        out: dict[str, dict[str, float]] = {}
        for t, rs in groups.items():
            n = len(rs)
            moved = sum(r.verdict != "unchanged" for r in rs)
            dprob = sum(r.prob_after - r.prob_before for r in rs) / n
            out[t] = {"n": n, "moved": moved / n, "mean_dprob": dprob}
        return out

    def summary(self) -> str:
        c = Counter(r.verdict for r in self.rows)
        head = (
            f"{c['flipped']} flipped, {c['degraded']} degraded, "
            f"{c['improved']} improved, {c['unchanged']} unchanged  "
            f"(of {len(self.rows)})"
        )
        tags = self.by_tag()
        if not tags:
            return head
        lines = [
            f"    {t:<16} {d['flipped'] + d['degraded']:>2} hit / {sum(d.values()):>2}"
            for t, d in sorted(tags.items())
        ]
        return head + "\n  by tag:\n" + "\n".join(lines)

    def show(self, full: bool = False) -> str:
        """Print the report. Without `full`, only the rows that moved, plus a
        footer counting the ones held back. `.show(full=True)` prints every
        probe."""
        rows = sorted(self.rows, key=lambda r: (_ORDER[r.verdict], -r.impact))
        shown = rows if full else [r for r in rows if r.verdict != "unchanged"]
        body = []
        for r in shown:
            tag = f" [{','.join(r.tags)}]" if r.tags else ""
            body.append(
                f"  {_MARKS[r.verdict]} {r.prompt!r}{tag}\n"
                f"      {r.top1_before!r} -> {r.top1_after!r}   "
                f"target p {r.prob_before:.3f}->{r.prob_after:.3f}  "
                f"rank {r.rank_before}->{r.rank_after}  [{r.verdict}]"
            )
        hidden = len(rows) - len(shown)
        footer = (
            f"\n  ... {hidden} unchanged (call .show(full=True) to list them)"
            if hidden and not full
            else ""
        )
        out = self.summary() + "\n" + ("\n".join(body) if body else "  (nothing moved)") + footer
        print(out)
        return out


def diff_battery(before: BatteryResult, after: BatteryResult) -> BatteryDiff:
    bp, ap = before.by_prompt(), after.by_prompt()
    rows = []
    for prompt, b in bp.items():
        a = ap[prompt]
        rows.append(
            DiffRow(
                prompt=prompt,
                tags=b.tags,
                top1_before=b.top1,
                top1_after=a.top1,
                prob_before=b.target_prob,
                prob_after=a.target_prob,
                rank_before=b.target_rank,
                rank_after=a.target_rank,
                verdict=_verdict(b, a),
            )
        )
    return BatteryDiff(rows)


def study_edit(model, tokenizer, intervention, battery, device: str = "cpu") -> BatteryDiff:
    """Run `battery` with and without `intervention` (a context manager, e.g.
    `marv.edit.suppress(model, feats)`), return the diff.

        rep = study_edit(model, tok, suppress(model, [(24, 4123)]), battery)
        rep.show()
    """
    before = run_battery(model, tokenizer, battery, device)
    with intervention:
        after = run_battery(model, tokenizer, battery, device)
    return diff_battery(before, after)


def suppression_frontier(model, tokenizer, constellation_feats, battery, sizes, device: str = "cpu"):
    """Sweep constellation size: for each n in `sizes`, suppress
    `constellation_feats[:n]` and diff the battery against the unedited
    baseline. Returns [(n, BatteryDiff), ...].

    Plot `metrics()['<target tag>']['mean_dprob']` (efficacy) against a
    neighbour/control tag's (collateral) across n to get the edit's Pareto
    frontier -- the "how hard can I push before I break the neighbours"
    curve that every knowledge-editing method lives or dies on.
    """
    from .edit import suppress

    base = run_battery(model, tokenizer, battery, device)
    out = []
    for n in sizes:
        with suppress(model, list(constellation_feats)[:n]):
            after = run_battery(model, tokenizer, battery, device)
        out.append((n, diff_battery(base, after)))
    return out
