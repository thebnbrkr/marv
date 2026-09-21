"""Tracing: which layers, and which of their two parts, produced a prediction,
and where information travels through depth.

Ported from marv-hyena (the Evo 2 fork), where these tools found that a single
late block carried the whole prediction. Llama-style decoder layers add exactly
two terms to the residual stream:

    h = h + self_attn(input_layernorm(h))          # the ATTENTION write
    h = h + mlp(post_attention_layernorm(h))       # the MLP write

so   final = embedding + sum over layers of (attn_write + mlp_write)
and  logits = lm_head(norm(final)).

Two kinds of answer, keep them apart:

- DIRECT (decompose_logit): each write's own push on a logit difference,
  through the final RMSNorm with its denominator frozen at the value it
  actually took. The rows add up EXACTLY to the model's real logit
  difference, and every result carries that check (`actual` vs `total`). A
  write that matters only by changing later layers shows up small here.
- TOTAL (patch_sweep, trace_by_depth, mean_ablate): swap or remove a
  component in a real forward pass and measure the output. Includes every
  indirect path.

Lesson from Evo 2: a correct direct attribution can still be useless. If one
layer's write dominates the residual (see diagnostics.find_bottlenecks), the
direct answer is "that layer, 100%" whatever the real mechanism is. Check for
bottlenecks before trusting a direct trace.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch

PARTS = ("attn", "mlp")
Component = tuple[int, str]  # (layer, "attn" | "mlp")


# ------------------------------------------------------------------ structure
def _layers(model):
    return model.model.layers


def num_layers(model) -> int:
    return len(_layers(model))


def component(model, layer: int, part: str):
    """The module whose forward OUTPUT is exactly this component's residual write."""
    blk = _layers(model)[layer]
    if part == "attn":
        return blk.self_attn
    if part == "mlp":
        return blk.mlp
    raise ValueError(f"part must be one of {PARTS}, got {part!r}")


def all_components(model, parts=PARTS) -> list[Component]:
    return [(i, p) for i in range(num_layers(model)) for p in parts]


def _first(output):
    return output[0] if isinstance(output, tuple) else output


def _rebuild(output, new):
    return (new, *output[1:]) if isinstance(output, tuple) else new


def _ids(tokenizer, prompt: str, device: str):
    enc = tokenizer(prompt, return_tensors="pt")
    return enc["input_ids"].to(device)


# ------------------------------------------------------------------ hooks
@contextmanager
def replace_outputs(model, replacements: dict[Component, Callable[[torch.Tensor], torch.Tensor]]):
    """Replace components' residual writes during the `with` block. Each value
    maps the original write (batch, seq, hidden) to its replacement. Hooks are
    removed on exit even if the forward pass raises."""
    handles = []

    def make(fn):
        def hook(_m, _i, output):
            x = _first(output)
            return _rebuild(output, fn(x).to(dtype=x.dtype, device=x.device))
        return hook

    try:
        for comp, fn in replacements.items():
            handles.append(component(model, *comp).register_forward_hook(make(fn)))
        yield
    finally:
        for h in handles:
            h.remove()


@dataclass
class Writes:
    """Every residual write at the captured positions, float32 on CPU."""

    positions: list[int]
    embed: torch.Tensor  # (P, H)
    parts: dict[Component, torch.Tensor]  # (P, H) each
    final: torch.Tensor  # (P, H): the residual entering the final norm
    logits: torch.Tensor  # (P, vocab)

    def reconstruction_error(self) -> float:
        """Relative error of embed + sum(writes) vs the real final residual.
        ~0 in float32. A large value means a write is being missed (e.g. an
        architecture with extra norms that isn't really Llama-style)."""
        total = self.embed + sum(self.parts.values())
        return float((total - self.final).norm() / self.final.norm().clamp_min(1e-12))


@torch.no_grad()
def capture_writes(model, tokenizer, prompt: str, positions: list[int] | None = None,
                   device: str = "cpu") -> Writes:
    """Capture the embedding, every attention and MLP write, and the final
    residual at `positions` (default: the last token)."""
    ids = _ids(tokenizer, prompt, device)
    positions = [ids.shape[1] - 1] if positions is None else list(positions)
    store: dict = {}
    handles = []

    def grab(key):
        def hook(_m, _i, out):
            store[key] = _first(out)[0, positions].detach().float().cpu()
        return hook

    def grab_final(_m, inputs):
        store["final"] = inputs[0][0, positions].detach().float().cpu()

    try:
        handles.append(model.model.embed_tokens.register_forward_hook(grab("embed")))
        handles.append(model.model.norm.register_forward_pre_hook(grab_final))
        for c in all_components(model):
            handles.append(component(model, *c).register_forward_hook(grab(c)))
        logits = model(input_ids=ids).logits
    finally:
        for h in handles:
            h.remove()
    parts = {c: store[c] for c in all_components(model)}
    return Writes(positions, store["embed"], parts, store["final"], logits[0, positions].float().cpu())


# ------------------------------------------------------------------ direct attribution
@dataclass
class Row:
    layer: int | None  # None = embedding
    part: str  # attn | mlp | embed
    value: float


@dataclass
class Decomposition:
    label: str
    rows: list[Row]
    actual: float  # read from the model's real logits
    extras: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(r.value for r in self.rows)

    @property
    def check_error(self) -> float:
        """|sum of rows - real value|, relative. ~0 in float32; if not, don't trust the rows."""
        return abs(self.total - self.actual) / max(abs(self.actual), 1e-9)

    def by_part(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for r in self.rows:
            out[r.part] += r.value
        return dict(out)

    def by_layer(self) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for r in self.rows:
            if r.layer is not None:
                out[r.layer] += r.value
        return dict(out)

    def top(self, k: int = 10) -> list[Row]:
        return sorted(self.rows, key=lambda r: -abs(r.value))[:k]

    def show(self, k: int = 10) -> str:
        lines = [f"{self.label}: actual={self.actual:+.3f}  sum-of-rows={self.total:+.3f}  "
                 f"(check error {self.check_error:.1e})",
                 "  by part: " + "  ".join(f"{a}={b:+.3f}" for a, b in sorted(self.by_part().items()))]
        for r in self.top(k):
            where = "embed" if r.layer is None else f"L{r.layer:<3} {r.part}"
            lines.append(f"    {where:<10} {r.value:+.3f}")
        out = "\n".join(lines)
        print(out)
        return out


def _token_id(tokenizer, token) -> int:
    if isinstance(token, int):
        return token
    from .evaluate import _target_token_ids

    return _target_token_ids(tokenizer, token)[0]


def decompose_logit(model, tokenizer, prompt: str, target, baseline=None, device: str = "cpu") -> Decomposition:
    """Split logit[target] - logit[baseline] at the last position into the
    direct push of the embedding and of every attention and MLP write.
    `baseline=None` uses the mean logit over the vocabulary. target/baseline
    may be strings (first token of the continuation) or token ids.

        d = decompose_logit(model, tok, "The capital of France is", "Paris")
        d.show()          # which layers pushed "Paris" up, and by how much
        d.check_error     # ~0: the rows add up to the real logit difference
    """
    w = capture_writes(model, tokenizer, prompt, device=device)
    W = model.lm_head.weight.detach().float().cpu()  # (vocab, H)
    t = _token_id(tokenizer, target)
    tw = torch.zeros(W.shape[0])
    tw[t] = 1.0
    if baseline is None:
        tw -= 1.0 / W.shape[0]
        label = f"logit[{target!r}] - mean logit"
    else:
        tw[_token_id(tokenizer, baseline)] -= 1.0
        label = f"logit[{target!r}] - logit[{baseline!r}]"
    direction = tw @ W  # (H,)

    norm = model.model.norm
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    x = w.final[0]
    s = norm.weight.detach().float().cpu() / torch.sqrt(x.pow(2).mean() + eps)  # frozen RMSNorm factor
    d = s * direction

    rows = [Row(None, "embed", float(w.embed[0] @ d))]
    rows += [Row(layer, part, float(v[0] @ d)) for (layer, part), v in sorted(w.parts.items())]
    actual = float(w.logits[0] @ tw)
    return Decomposition(label, rows, actual, extras={"reconstruction_error": w.reconstruction_error()})


# ------------------------------------------------------------------ total effects
@torch.no_grad()
def mean_writes(model, tokenizer, texts: list[str], components: list[Component], device: str = "cpu"):
    """Mean write of each component over every position of `texts`
    (reduced inside the hook, so memory stays O(hidden))."""
    sums: dict[Component, torch.Tensor] = {}
    count = 0
    handles = []

    def make(c):
        def hook(_m, _i, out):
            x = _first(out)[0].float()
            sums[c] = sums.get(c, 0) + x.sum(0)
        return hook

    try:
        for c in components:
            handles.append(component(model, *c).register_forward_hook(make(c)))
        for text in texts:
            ids = _ids(tokenizer, text, device)
            count += ids.shape[1]
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    return {c: v / count for c, v in sums.items()}


@contextmanager
def mean_ablate(model, means: dict[Component, torch.Tensor]):
    """Replace each component's write with its mean write (from mean_writes).
    Prefer this to zeroing: zeroing pushes the residual off-distribution and
    often just breaks the model, which then fails every test at once."""
    fns = {c: (lambda mu: (lambda x: mu.to(x.device, x.dtype).expand_as(x).clone()))(mu) for c, mu in means.items()}
    with replace_outputs(model, fns):
        yield


def logit_diff_metric(tokenizer, target, baseline=None):
    """metric(logits) = logit[target] - logit[baseline or mean] at the last position."""
    t = _token_id(tokenizer, target)
    b = None if baseline is None else _token_id(tokenizer, baseline)

    def metric(logits):
        last = logits[0, -1].float()
        return float(last[t] - (last.mean() if b is None else last[b]))
    return metric


@dataclass
class PatchResult:
    component: Component | str
    metric: float
    fraction: float  # (patched - target_run) / (source_run - target_run)


@dataclass
class PatchSweep:
    target_metric: float
    source_metric: float
    results: list[PatchResult]

    def top(self, k: int = 10) -> list[PatchResult]:
        return sorted(self.results, key=lambda r: -abs(r.fraction))[:k]

    def show(self, k: int = 10) -> str:
        lines = [f"target run={self.target_metric:+.3f}  source run={self.source_metric:+.3f}"]
        for r in self.top(k):
            name = r.component if isinstance(r.component, str) else f"L{r.component[0]} {r.component[1]}"
            lines.append(f"  {name:<14} restores {r.fraction:+7.1%}")
        out = "\n".join(lines)
        print(out)
        return out


@torch.no_grad()
def patch_sweep(model, tokenizer, source: str, target: str, metric, components: list[Component] | None = None,
                include_groups: bool = True, device: str = "cpu") -> PatchSweep:
    """Copy each component's write from the SOURCE run into the TARGET run
    (same token length) and report the share of metric(source) - metric(target)
    it restores. That's a total effect, including indirect paths.

        m = logit_diff_metric(tok, " Rome", " Paris")
        patch_sweep(model, tok, "The capital of Italy is", "The capital of France is", m).show()
    """
    src_ids, tgt_ids = _ids(tokenizer, source, device), _ids(tokenizer, target, device)
    if src_ids.shape != tgt_ids.shape:
        raise ValueError("source and target must tokenize to the same length")
    components = all_components(model) if components is None else components
    cache: dict[Component, torch.Tensor] = {}
    handles = []

    def grab(c):
        def hook(_m, _i, out):
            cache[c] = _first(out).detach().clone()
        return hook

    try:
        for c in components:
            handles.append(component(model, *c).register_forward_hook(grab(c)))
        src_m = metric(model(input_ids=src_ids).logits)
    finally:
        for h in handles:
            h.remove()
    tgt_m = metric(model(input_ids=tgt_ids).logits)
    denom = src_m - tgt_m if abs(src_m - tgt_m) > 1e-9 else float("nan")

    def run(comps):
        with replace_outputs(model, {c: (lambda v: (lambda x: v))(cache[c]) for c in comps}):
            return metric(model(input_ids=tgt_ids).logits)

    results = []
    for c in components:
        m = run([c])
        results.append(PatchResult(c, m, (m - tgt_m) / denom))
    if include_groups:
        for part in PARTS:
            group = [c for c in components if c[1] == part]
            if group:
                m = run(group)
                results.append(PatchResult(f"all {part}", m, (m - tgt_m) / denom))
    return PatchSweep(tgt_m, src_m, results)


@torch.no_grad()
def trace_by_depth(model, tokenizer, source: str, target: str, position: int, metric,
                   device: str = "cpu") -> list[dict]:
    """At which layer does information LEAVE `position`? For each depth
    (-1 = the token embedding, d = after layer d), copy the SOURCE run's whole
    residual at `position` into the TARGET run and measure metric.

    If source and target differ only at `position`, depth -1 reproduces the
    source run exactly (fraction 1). As depth grows, later positions have
    already read the target's residual at `position` in earlier layers, so the
    fraction falls where the model has moved the information elsewhere.
    In marv-hyena, Evo 2 moved a mutation's effect off its position within the
    first ~8 blocks. The same question for an LM: when does "France" hand its
    information to the answer position?
    """
    src_ids, tgt_ids = _ids(tokenizer, source, device), _ids(tokenizer, target, device)
    if src_ids.shape != tgt_ids.shape:
        raise ValueError("source and target must tokenize to the same length")
    mods = [(-1, model.model.embed_tokens)] + [(d, blk) for d, blk in enumerate(_layers(model))]
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def grab(d):
        def hook(_m, _i, out):
            captured[d] = _first(out)[0, position].detach().clone()
        return hook

    try:
        for d, m in mods:
            handles.append(m.register_forward_hook(grab(d)))
        src_m = metric(model(input_ids=src_ids).logits)
    finally:
        for h in handles:
            h.remove()
    tgt_m = metric(model(input_ids=tgt_ids).logits)
    denom = src_m - tgt_m if abs(src_m - tgt_m) > 1e-9 else float("nan")

    rows = []
    for d, m in mods:
        def patch(_m, _i, out, d=d):
            x = _first(out).clone()
            x[0, position] = captured[d].to(x.device, x.dtype)
            return _rebuild(out, x)

        h = m.register_forward_hook(patch)
        try:
            pm = metric(model(input_ids=tgt_ids).logits)
        finally:
            h.remove()
        rows.append({"depth": d, "metric": pm, "fraction": (pm - tgt_m) / denom})
    return rows
