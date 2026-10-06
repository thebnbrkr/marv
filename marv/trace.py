"""Tracing: which layers, and which of their two parts, produced a prediction,
and where information travels through depth.

Ported from marv-hyena (the Evo 2 fork), where these tools found that a single
late block carried the whole prediction. Llama-style decoder layers add exactly
two terms to the residual stream:

    h = h + self_attn(input_layernorm(h))          # the ATTENTION write
    h = h + mlp(post_attention_layernorm(h))       # the MLP write

so   final = embedding + sum over layers of (attn_write + mlp_write)
and  logits = lm_head(norm(final)).

Every module lookup goes through the model's architecture adapter
(marv.arch), so other layouts work unchanged as long as their layers only
ADD writes to the residual: a Whisper decoder layer adds three
(self-attention, cross-attention to the audio, MLP), and its final norm is a
LayerNorm, which contributes a constant bias row of its own. `prompt`
arguments accept a dict of model inputs in place of a string, for models
that take more than token ids.

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
import torch.nn as nn

PARTS = ("attn", "mlp")  # Llama-style; other adapters name their own
Component = tuple[int, str]  # (layer, part), e.g. (3, "mlp")


# ------------------------------------------------------------------ structure
def _adapter(model):
    from .arch import detect_adapter

    return detect_adapter(model)


def _layers(model):
    return _adapter(model).layers(model)


def num_layers(model) -> int:
    return len(_layers(model))


def parts_of(model) -> tuple[str, ...]:
    """The residual writes each layer of this model makes, in forward order."""
    return _adapter(model).parts


def component(model, layer: int, part: str):
    """The module whose forward OUTPUT is exactly this component's residual write."""
    return _adapter(model).component(model, layer, part)


def all_components(model, parts=None) -> list[Component]:
    parts = parts_of(model) if parts is None else parts
    return [(i, p) for i in range(num_layers(model)) for p in parts]


def _first(output):
    return output[0] if isinstance(output, tuple) else output


def _rebuild(output, new):
    return (new, *output[1:]) if isinstance(output, tuple) else new


def _inputs(model, tokenizer, prompt, device: str) -> dict:
    """Model kwargs for `prompt`: a string is tokenized, a dict passes through."""
    return _adapter(model).inputs(tokenizer, prompt, device)


def _same_shape(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        getattr(a[k], "shape", None) == getattr(b[k], "shape", None) for k in a)


def _hidden_arg(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def _with_hidden(args, kwargs, new):
    if args:
        return (new, *args[1:]), kwargs
    return args, {**kwargs, "hidden_states": new}


def _frozen_norm(norm, x: torch.Tensor):
    """The final norm with its statistics frozen at the values they took on
    the real residual `x` (H,). Returns (f, const): norm(x) == f(x) + const,
    and f is linear, so f(write) is each write's exact share.

    RMSNorm: w * x / rms(x), no constant. LayerNorm: g * (x - mean) / s + b,
    where each write is centred on its own mean (the means add up to x's) and
    the bias b is a constant that belongs to no write."""
    w = norm.weight.detach().float().cpu()
    if isinstance(norm, nn.LayerNorm):
        s = torch.sqrt(x.var(unbiased=False) + norm.eps)
        b = norm.bias.detach().float().cpu() if norm.bias is not None else torch.zeros_like(w)
        return (lambda v: w * (v - v.mean(-1, keepdim=True)) / s), b
    if type(norm).__name__.endswith("RMSNorm"):
        eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
        r = torch.sqrt(x.pow(2).mean() + eps)
        return (lambda v: w * v / r), torch.zeros_like(w)
    raise TypeError(f"no exact frozen form for final norm {type(norm).__name__}")


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
def capture_writes(model, tokenizer, prompt, positions: list[int] | None = None,
                   device: str = "cpu") -> Writes:
    """Capture the embedding (the residual entering the first layer), every
    layer's writes, and the final residual at `positions` (default: the last
    token)."""
    ad = _adapter(model)
    inputs = _inputs(model, tokenizer, prompt, device)
    store: dict = {}
    handles = []

    def at(x):
        return x[0, [x.shape[1] - 1] if positions is None else list(positions)].detach().float().cpu()

    def grab(key):
        def hook(_m, _i, out):
            store[key] = at(_first(out))
        return hook

    def grab_embed(_m, args, kwargs):
        store["embed"] = at(_hidden_arg(args, kwargs))

    def grab_final(_m, args):
        store["final"] = at(args[0])

    comps = all_components(model)
    try:
        handles.append(ad.layers(model)[0].register_forward_pre_hook(grab_embed, with_kwargs=True))
        handles.append(ad.final_norm(model).register_forward_pre_hook(grab_final))
        for c in comps:
            handles.append(ad.component(model, *c).register_forward_hook(grab(c)))
        logits = model(**inputs).logits
    finally:
        for h in handles:
            h.remove()
    pos = [logits.shape[1] - 1] if positions is None else list(positions)
    parts = {c: store[c] for c in comps}
    return Writes(pos, store["embed"], parts, store["final"], logits[0, pos].float().cpu())


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


def decompose_logit(model, tokenizer, prompt, target, baseline=None, device: str = "cpu") -> Decomposition:
    """Split logit[target] - logit[baseline] at the last position into the
    direct push of the embedding and of every layer's writes (plus, for a
    LayerNorm, the norm's constant bias row).
    `baseline=None` uses the mean logit over the vocabulary. target/baseline
    may be strings (first token of the continuation) or token ids.

        d = decompose_logit(model, tok, "The capital of France is", "Paris")
        d.show()          # which layers pushed "Paris" up, and by how much
        d.check_error     # ~0: the rows add up to the real logit difference
    """
    ad = _adapter(model)
    w = capture_writes(model, tokenizer, prompt, device=device)
    W = ad.unembed(model).float().cpu()  # (vocab, H)
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

    norm = ad.final_norm(model)
    f, const = _frozen_norm(norm, w.final[0])

    rows = [Row(None, "embed", float(f(w.embed[0]) @ direction))]
    rows += [Row(layer, part, float(f(v[0]) @ direction)) for (layer, part), v in sorted(w.parts.items())]
    if isinstance(norm, nn.LayerNorm):
        rows.append(Row(None, "bias", float(const @ direction)))
    actual = float(w.logits[0] @ tw)
    return Decomposition(label, rows, actual, extras={"reconstruction_error": w.reconstruction_error()})


# ------------------------------------------------------------------ total effects
@torch.no_grad()
def mean_writes(model, tokenizer, texts: list, components: list[Component], skip_first: bool = True,
                device: str = "cpu"):
    """Mean write of each component over the positions of `texts` (reduced
    inside the hook, so memory stays O(hidden)).

    Position 0 is skipped by default: the first token in most language
    models carries huge, atypical activations (an attention sink), and
    averaging it in skews the mean that mean_ablate substitutes."""
    sums: dict[Component, torch.Tensor] = {}
    counts: dict[Component, int] = {}
    handles = []

    def make(c):
        def hook(_m, _i, out):
            x = _first(out)[0].float()
            if skip_first and x.shape[0] > 1:
                x = x[1:]
            sums[c] = sums.get(c, 0) + x.sum(0)
            counts[c] = counts.get(c, 0) + x.shape[0]
        return hook

    try:
        for c in components:
            handles.append(component(model, *c).register_forward_hook(make(c)))
        for text in texts:
            model(**_inputs(model, tokenizer, text, device))
    finally:
        for h in handles:
            h.remove()
    return {c: v / counts[c] for c, v in sums.items()}


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
def patch_sweep(model, tokenizer, source, target, metric, components: list[Component] | None = None,
                include_groups: bool = True, device: str = "cpu") -> PatchSweep:
    """Copy each component's write from the SOURCE run into the TARGET run
    (same token length) and report the share of metric(source) - metric(target)
    it restores. That's a total effect, including indirect paths.

        m = logit_diff_metric(tok, " Rome", " Paris")
        patch_sweep(model, tok, "The capital of Italy is", "The capital of France is", m).show()
    """
    src, tgt = _inputs(model, tokenizer, source, device), _inputs(model, tokenizer, target, device)
    if not _same_shape(src, tgt):
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
        src_m = metric(model(**src).logits)
    finally:
        for h in handles:
            h.remove()
    tgt_m = metric(model(**tgt).logits)
    denom = src_m - tgt_m if abs(src_m - tgt_m) > 1e-9 else float("nan")

    def run(comps):
        with replace_outputs(model, {c: (lambda v: (lambda x: v))(cache[c]) for c in comps}):
            return metric(model(**tgt).logits)

    results = []
    for c in components:
        m = run([c])
        results.append(PatchResult(c, m, (m - tgt_m) / denom))
    if include_groups:
        for part in parts_of(model):
            group = [c for c in components if c[1] == part]
            if group:
                m = run(group)
                results.append(PatchResult(f"all {part}", m, (m - tgt_m) / denom))
    return PatchSweep(tgt_m, src_m, results)


@torch.no_grad()
def trace_by_depth(model, tokenizer, source, target, position: int, metric,
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
    src, tgt = _inputs(model, tokenizer, source, device), _inputs(model, tokenizer, target, device)
    if not _same_shape(src, tgt):
        raise ValueError("source and target must tokenize to the same length")
    layers = _layers(model)
    depths = [-1] + list(range(len(layers)))
    captured: dict[int, torch.Tensor] = {}
    handles = []

    # depth -1 is the residual entering the first layer; depth d is layer d's output
    def grab_in(_m, args, kwargs):
        captured[-1] = _hidden_arg(args, kwargs)[0, position].detach().clone()

    def grab(d):
        def hook(_m, _i, out):
            captured[d] = _first(out)[0, position].detach().clone()
        return hook

    def patcher(d):
        if d == -1:
            def pre(_m, args, kwargs):
                x = _hidden_arg(args, kwargs).clone()
                x[0, position] = captured[-1].to(x.device, x.dtype)
                return _with_hidden(args, kwargs, x)
            return layers[0].register_forward_pre_hook(pre, with_kwargs=True)

        def post(_m, _i, out):
            x = _first(out).clone()
            x[0, position] = captured[d].to(x.device, x.dtype)
            return _rebuild(out, x)
        return layers[d].register_forward_hook(post)

    try:
        handles.append(layers[0].register_forward_pre_hook(grab_in, with_kwargs=True))
        for d, blk in enumerate(layers):
            handles.append(blk.register_forward_hook(grab(d)))
        src_m = metric(model(**src).logits)
    finally:
        for h in handles:
            h.remove()
    tgt_m = metric(model(**tgt).logits)
    denom = src_m - tgt_m if abs(src_m - tgt_m) > 1e-9 else float("nan")

    rows = []
    for d in depths:
        h = patcher(d)
        try:
            pm = metric(model(**tgt).logits)
        finally:
            h.remove()
        rows.append({"depth": d, "metric": pm, "fraction": (pm - tgt_m) / denom})
    return rows
