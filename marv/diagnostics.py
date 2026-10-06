"""Diagnostics: checks to run BEFORE trusting an edit or a trace.

Ported from marv-hyena, where each one fixed a real mistake on Evo 2:

- health          Does the model still work under an intervention? Stubbing
                  a component the whole model depends on makes EVERY test
                  fail at once. Read without a health check, that looks like
                  "this component does X" when it means "the model is broken".
- load_bearing    Which single layers break the model on their own? Group
                  ablations that include one of these only measure breakage;
                  keep them on (e.g. mean-ablate a band minus its
                  load-bearing layers).
- write_norms /   Is one layer's write so large that it dominates the residual?
  find_bottlenecks Then direct attribution says "that layer, 100%" whatever
                  the mechanism. Evo 2's block 30 wrote ~1e12 vs <=1e7 for
                  every other block.
- dead_features   Neurons that never fire on the reference text. Their
                  "top examples" are noise; drop them from describe-style
                  summaries. Evo 2's first layer had 18% dead channels.
- null_model      The same weights, shuffled inside each tensor: identical
                  value statistics, no learned structure. Run the same
                  analysis on it; anything that also shows up there is an
                  artifact of the method or architecture, not something learned.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from .trace import _adapter, _inputs, all_components, capture_writes, mean_ablate, mean_writes


# ------------------------------------------------------------------ health
@dataclass
class Health:
    accuracy: float  # next-token top-1 accuracy over the reference texts
    logprob: float  # mean log-prob of the true next token

    def broken_vs(self, baseline: "Health", keep: float = 0.5) -> bool:
        """Broken if accuracy fell below `keep` x the baseline's."""
        return self.accuracy < keep * baseline.accuracy


@torch.no_grad()
def health(model, tokenizer, texts: list[str], device: str = "cpu") -> Health:
    """Next-token accuracy and mean log-prob over ordinary text. Run it with
    and without an intervention (or pass health_texts= to study_edit)."""
    correct = total = 0
    lp_sum = 0.0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        if ids.shape[1] < 2:
            continue
        logits = model(input_ids=ids).logits[0, :-1].float()
        tgt = ids[0, 1:]
        lp = torch.log_softmax(logits, -1).gather(-1, tgt[:, None])[:, 0]
        correct += int((logits.argmax(-1) == tgt).sum())
        total += tgt.numel()
        lp_sum += float(lp.sum())
    if total == 0:
        raise ValueError("health() needs texts at least 2 tokens long")
    return Health(correct / total, lp_sum / total)


# ------------------------------------------------------------------ load-bearing map
@dataclass
class LoadBearingRow:
    layer: int
    part: str
    health: Health
    broken: bool


def load_bearing(model, tokenizer, texts: list[str], parts=None, keep: float = 0.5,
                 device: str = "cpu") -> list[LoadBearingRow]:
    """Mean-ablate each single component (layer, part) and report health.
    `broken` = accuracy below `keep` x the unablated model's."""
    base = health(model, tokenizer, texts, device)
    comps = all_components(model, parts)
    means = mean_writes(model, tokenizer, texts, comps, device)
    rows = []
    for c in comps:
        with mean_ablate(model, {c: means[c]}):
            h = health(model, tokenizer, texts, device)
        rows.append(LoadBearingRow(c[0], c[1], h, h.broken_vs(base, keep)))
    return rows


# ------------------------------------------------------------------ bottlenecks
@dataclass
class WriteNorm:
    layer: int | None  # None = embedding
    part: str
    norm: float  # mean L2 norm of the write at the captured positions
    share: float  # norm / norm of the final residual


def write_norms(model, tokenizer, prompt, positions: list[int] | None = None,
                device: str = "cpu") -> list[WriteNorm]:
    """Size of every write to the residual stream, and its share of the final residual."""
    w = capture_writes(model, tokenizer, prompt, positions, device)
    final = float(w.final.norm(dim=-1).mean())
    rows = [WriteNorm(None, "embed", float(w.embed.norm(dim=-1).mean()), 0.0)]
    rows += [WriteNorm(l, p, float(x.norm(dim=-1).mean()), 0.0) for (l, p), x in sorted(w.parts.items())]
    for r in rows:
        r.share = r.norm / max(final, 1e-12)
    return rows


def find_bottlenecks(model, tokenizer, prompt, positions: list[int] | None = None,
                     min_share: float = 0.5, device: str = "cpu") -> list[tuple[int, str]]:
    """Components whose write alone is >= min_share of the final residual's norm."""
    return [(r.layer, r.part) for r in write_norms(model, tokenizer, prompt, positions, device)
            if r.layer is not None and r.share >= min_share]


# ------------------------------------------------------------------ dead features
@torch.no_grad()
def dead_features(model, tokenizer, texts: list[str], tol: float = 1e-6,
                  device: str = "cpu") -> dict[int, np.ndarray]:
    """Per layer, indices of MLP neurons (MARV features) whose activation
    act(gate) * up never exceeds `tol` in magnitude on any token of `texts`."""
    ad = _adapter(model)
    layers = ad.layers(model)
    peak = [None] * len(layers)
    handles = []

    def make(i):
        def pre(_m, args):
            a = args[0][0].abs().amax(0).float()
            peak[i] = a if peak[i] is None else torch.maximum(peak[i], a)
        return pre

    try:
        for i in range(len(layers)):
            handles.append(ad.ffn_out(model, i).register_forward_pre_hook(make(i)))
        for text in texts:
            model(**_inputs(model, tokenizer, text, device))
    finally:
        for h in handles:
            h.remove()
    return {i: torch.nonzero(p <= tol).flatten().cpu().numpy() for i, p in enumerate(peak)}


# ------------------------------------------------------------------ null model
def null_model(model, seed: int = 0):
    """A copy of `model` whose every weight tensor is randomly permuted: the
    same value distribution per tensor, none of the learned structure. Use it
    as a baseline: run your analysis on both, and treat whatever survives on
    the null model as an artifact. Doubles memory (it's a deep copy)."""
    null = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in null.parameters():
            flat = p.data.reshape(-1)
            idx = torch.randperm(flat.numel(), generator=g).to(flat.device)
            p.data.copy_(flat[idx].reshape(p.shape))
    return null
