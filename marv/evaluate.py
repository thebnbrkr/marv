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
    target: str  # expected continuation; scored on its first token
    tags: tuple[str, ...] = ()
    # explicit candidate first-token ids for the target. If None, derived
    # from `target` with/without a leading space (see _target_token_ids) --
    # BPE tokenizers emit a different id for " Paris" (mid-sentence) than
    # "Paris" (start), and scoring the wrong one makes a known fact look
    # rank-1000. Set this when the auto-derivation is still wrong.
    target_ids: tuple[int, ...] | None = None


@dataclass
class ProbeRow:
    prompt: str
    target: str
    target_ids: tuple[int, ...]  # candidate first-token ids for the target
    top1: str
    top1_id: int
    target_rank: int  # best (lowest) rank across target_ids
    target_prob: float  # summed probability across target_ids
    tags: tuple[str, ...]

    @property
    def target_id(self) -> int:  # back-compat: the best single candidate
        return self.target_ids[0]


@dataclass
class BatteryResult:
    rows: list[ProbeRow]

    def by_prompt(self) -> dict[str, ProbeRow]:
        return {r.prompt: r for r in self.rows}


def _target_token_ids(tokenizer, text: str) -> tuple[int, ...]:
    """Candidate first-token ids for a target continuation. BPE tokenizers
    emit a different id for ' Paris' (what the model actually predicts
    mid-sentence) than 'Paris' (sentence start), and the capitalised vs
    lower forms differ again -- so gather all plausible first tokens and let
    run_battery score the best/summed. Deduped, order-preserving."""
    out: list[int] = []
    forms = [" " + text, text, " " + text.strip(), text.strip()]
    forms += [" " + text.strip().lower(), text.strip().capitalize()]
    for form in forms:
        ids = tokenizer.encode(form, add_special_tokens=False)
        if ids and ids[0] not in out:
            out.append(int(ids[0]))
    if not out:
        raise ValueError(f"target {text!r} tokenized to nothing")
    return tuple(out)


@torch.no_grad()
def run_battery(model, tokenizer, probes, device: str = "cpu") -> BatteryResult:
    """Forward each probe once; record the top-1 next token, and the target's
    best rank + summed probability across its candidate first-token ids
    (see _target_token_ids)."""
    model.eval()
    rows: list[ProbeRow] = []
    for p in probes:
        enc = tokenizer(p.prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        order = torch.argsort(logits, descending=True)
        rank_of = {int(t): i for i, t in enumerate(order.tolist())}

        tids = tuple(p.target_ids) if p.target_ids else _target_token_ids(tokenizer, p.target)
        tids = tuple(t for t in tids if t in rank_of) or tids
        rank = min(rank_of.get(t, len(order) - 1) for t in tids) + 1
        prob = float(sum(probs[t] for t in tids))
        top1_id = int(order[0])
        rows.append(
            ProbeRow(
                prompt=p.prompt,
                target=p.target,
                target_ids=tids,
                top1=tokenizer.decode([top1_id]).strip(),
                top1_id=top1_id,
                target_rank=rank,
                target_prob=prob,
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


def _verdict(b: ProbeRow, a: ProbeRow, eps: float = 0.05, rel: float = 0.5) -> str:
    """Target-centric: does the *target token* gain or lose ground. A control
    where the model was already wrong and its top-1 wanders does not count as
    an edit effect unless the target's own rank/prob moved.

    - flipped  : target held rank 1 and lost it, or gained rank 1
    - degraded : target prob fell by >= eps absolute or >= rel fraction
    - improved : target prob rose by >= eps absolute or >= rel fraction
    """
    lost_top1 = b.target_rank == 1 and a.target_rank != 1
    gained_top1 = b.target_rank != 1 and a.target_rank == 1
    if lost_top1 or gained_top1:
        return "flipped"
    dp = a.target_prob - b.target_prob
    denom = max(b.target_prob, 1e-9)
    if dp < -eps or dp / denom < -rel:
        return "degraded"
    if dp > eps or dp / denom > rel:
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


def rank_by_ablation_effect(model, tokenizer, candidates, probes, device: str = "cpu", restore_between: bool = True):
    """Rank `candidates` (a list of (layer, feature)) by how much suppressing
    each ONE, alone, drops the mean target probability across `probes`. The
    *causal* constellation -- it measures the thing you actually want (effect
    on the fact) instead of a geometric proxy (gate-KNN similarity).

    One forward pass per candidate per probe, so keep the candidate pool
    modest (e.g. constellation()[:30]) and `probes` to the target rephrasings.
    Returns [((layer, feature), mean_prob_drop), ...] sorted by drop desc.
    """
    from .edit import suppress

    base = {r.prompt: r.target_prob for r in run_battery(model, tokenizer, probes, device).rows}
    scored = []
    for c in candidates:
        with suppress(model, [tuple(c)]):
            after = run_battery(model, tokenizer, probes, device).rows
        drop = sum(base[r.prompt] - r.target_prob for r in after) / len(after)
        scored.append((tuple(c), drop))
    scored.sort(key=lambda x: -x[1])
    return scored


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
