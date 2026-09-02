"""vindex extraction: pull gate/down features out of a loaded model into
plain numpy, keyed by (layer, feature_index) -- the unit MARV operates on.

No mmap, no on-disk format, no streaming -- a 135M/1.7B model fits in RAM
whole, so this just walks the loaded torch model and copies tensors to numpy.
`extract_streaming()` covers the case where the checkpoint does *not* fit:
it reads one layer's tensors at a time straight from the safetensors files
and never instantiates the torch model (so no contextual probing).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .arch import ArchAdapter, detect_adapter


def default_layer_bands(num_layers: int) -> dict[str, tuple[int, int]]:
    """Heuristic syntax / knowledge / output split, as fractions of depth.

    Mirrors the per-family bands LARQL auto-detects (Gemma 3 34L: 0-13 /
    14-27 / 28-33; Llama 3 32L: 0-7 / 8-24 / 25-31): early layers do
    token/morphology work, the middle band carries factual associations,
    the last sliver formats the output. These are boundaries for *where to
    look*, not hard architectural lines -- override `VindexLite.layer_bands`
    if a per-prompt heatmap says otherwise for your model.
    """
    if num_layers <= 3:
        # not enough depth for three disjoint bands; degrade gracefully
        last = num_layers - 1
        return {
            "syntax": (0, 0),
            "knowledge": (min(1, last), max(1, last) if num_layers >= 3 else last),
            "output": (last, last),
        }
    syn_end = max(0, round(num_layers * 0.40) - 1)
    know_end = min(max(syn_end + 1, round(num_layers * 0.85) - 1), num_layers - 2)
    return {
        "syntax": (0, syn_end),
        "knowledge": (syn_end + 1, know_end),
        "output": (know_end + 1, num_layers - 1),
    }


@dataclass
class VindexLite:
    """All the gate rows and down columns for one model, plus what's needed
    to turn a down column back into token logits (logit lens)."""

    gate: list[np.ndarray]  # one (intermediate_size, hidden_size) array per layer
    down: list[np.ndarray]  # one (hidden_size, intermediate_size) array per layer
    embed: np.ndarray  # (vocab_size, hidden_size)
    lm_head: np.ndarray  # (vocab_size, hidden_size)
    hidden_size: int
    num_layers: int
    # Final RMSNorm's learned per-channel gain + eps. Required to turn a raw
    # residual-space vector into the same input the model's own lm_head
    # actually sees -- a plain "divide by RMS" skips this and the projection
    # ends up dominated by whichever hidden dims happen to have large raw
    # magnitude rather than by what the model's norm layer actually amplifies.
    final_norm_weight: np.ndarray | None = field(default=None)
    norm_eps: float = field(default=1e-6)
    model_name: str = field(default="")
    # Heuristic depth bands (see default_layer_bands). Set by extract().
    layer_bands: dict[str, tuple[int, int]] | None = field(default=None)
    # (layer, feature) pairs hidden from top_features / describe_*. A
    # retrieval-layer filter -- it does NOT change the weights or a forward
    # pass (see marv/edit.py for that). Reversible: discard the entry.
    suppressed: set[tuple[int, int]] = field(default_factory=set)
    # Optional precomputed logit-lens cache: per layer, (intermediate, k)
    # arrays of the top-k promoted token ids and their scores. Built by
    # marv.probe.build_down_meta(); makes describe_feature() a dict lookup
    # instead of a (vocab x hidden) matmul per call. Pure weight math -- no
    # forward pass, no attention.
    down_meta_tokens: list[np.ndarray] | None = field(default=None)
    down_meta_scores: list[np.ndarray] | None = field(default=None)

    def band(self, name: str) -> list[int]:
        """Layer indices in a named band ('syntax' | 'knowledge' | 'output')."""
        bands = self.layer_bands or default_layer_bands(self.num_layers)
        lo, hi = bands[name]
        return list(range(lo, hi + 1))

    def suppress(self, layer: int, feature: int) -> None:
        self.suppressed.add((int(layer), int(feature)))

    def unsuppress(self, layer: int, feature: int) -> None:
        self.suppressed.discard((int(layer), int(feature)))

    def save(self, path: str) -> None:
        extra: dict[str, np.ndarray] = {}
        if self.down_meta_tokens is not None and self.down_meta_scores is not None:
            for i, (t, s) in enumerate(zip(self.down_meta_tokens, self.down_meta_scores)):
                extra[f"dm_tok_{i}"] = t
                extra[f"dm_score_{i}"] = s
        suppressed = (
            np.array(sorted(self.suppressed), dtype=np.int64)
            if self.suppressed
            else np.zeros((0, 2), dtype=np.int64)
        )
        np.savez_compressed(
            path,
            embed=self.embed,
            lm_head=self.lm_head,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            final_norm_weight=self.final_norm_weight
            if self.final_norm_weight is not None
            else np.zeros(0, dtype=np.float32),
            norm_eps=self.norm_eps,
            model_name=self.model_name,
            layer_bands=json.dumps(self.layer_bands) if self.layer_bands else "",
            suppressed=suppressed,
            has_down_meta=self.down_meta_tokens is not None,
            **{f"gate_{i}": g for i, g in enumerate(self.gate)},
            **{f"down_{i}": d for i, d in enumerate(self.down)},
            **extra,
        )

    @classmethod
    def load(cls, path: str) -> "VindexLite":
        z = np.load(path, allow_pickle=False)
        n = int(z["num_layers"])
        norm_weight = z["final_norm_weight"] if "final_norm_weight" in z else np.zeros(0)

        layer_bands = None
        bands_raw = str(z["layer_bands"]) if "layer_bands" in z else ""
        if bands_raw:
            layer_bands = {k: tuple(v) for k, v in json.loads(bands_raw).items()}

        suppressed: set[tuple[int, int]] = set()
        if "suppressed" in z and z["suppressed"].size:
            suppressed = {(int(a), int(b)) for a, b in z["suppressed"]}

        dm_tok = dm_score = None
        if "has_down_meta" in z and bool(z["has_down_meta"]):
            dm_tok = [z[f"dm_tok_{i}"] for i in range(n)]
            dm_score = [z[f"dm_score_{i}"] for i in range(n)]

        return cls(
            gate=[z[f"gate_{i}"] for i in range(n)],
            down=[z[f"down_{i}"] for i in range(n)],
            embed=z["embed"],
            lm_head=z["lm_head"],
            hidden_size=int(z["hidden_size"]),
            num_layers=n,
            final_norm_weight=norm_weight if norm_weight.size else None,
            norm_eps=float(z["norm_eps"]) if "norm_eps" in z else 1e-6,
            model_name=str(z["model_name"]) if "model_name" in z else "",
            layer_bands=layer_bands,
            suppressed=suppressed,
            down_meta_tokens=dm_tok,
            down_meta_scores=dm_score,
        )


def extract(model, adapter: ArchAdapter | None = None, model_name: str = "") -> VindexLite:
    adapter = adapter or detect_adapter(model)
    n = adapter.num_layers(model)
    gate, down = [], []
    for i in range(n):
        layer = adapter.ffn_layer(model, i)
        # .numpy() is zero-copy when the tensor is already float32/CPU, which
        # would alias the model's live weight storage -- .copy() so a later
        # mutation on the model (or on another VindexLite view of the same
        # tensor) can never silently change an already-extracted vindex.
        gate.append(layer.gate.float().cpu().numpy().copy())
        down.append(layer.down.float().cpu().numpy().copy())
    embed = adapter.embed(model).float().cpu().numpy().copy()
    lm_head = adapter.lm_head(model).float().cpu().numpy().copy()
    final_norm_weight = adapter.final_norm(model).weight.detach().float().cpu().numpy().copy()
    norm_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))
    return VindexLite(
        gate=gate,
        down=down,
        embed=embed,
        lm_head=lm_head,
        hidden_size=embed.shape[1],
        num_layers=n,
        final_norm_weight=final_norm_weight,
        norm_eps=norm_eps,
        model_name=model_name or getattr(getattr(model, "config", None), "_name_or_path", ""),
        layer_bands=default_layer_bands(n),
    )


def extract_streaming(
    model_dir: str,
    *,
    num_layers: int | None = None,
    layer_key: str = "model.layers.{i}.mlp.{proj}_proj.weight",
    embed_key: str = "model.embed_tokens.weight",
    lm_head_key: str = "model.lm_head.weight",
    norm_key: str = "model.norm.weight",
    norm_eps: float = 1e-5,
    model_name: str = "",
) -> VindexLite:
    """Build a vindex from a directory of `.safetensors` shards without ever
    loading the whole model. Peak RAM ~= embeddings + one layer.

    Llama-style key names by default (`model.layers.{i}.mlp.gate_proj.weight`
    etc.). Point the `*_key` args elsewhere for a differently-named
    checkpoint. Needs the `safetensors` package; there is no live model
    afterwards, so `marv.context` contextual probing is unavailable on a
    vindex built this way -- use `extract(model)` for that.
    """
    import glob
    import os

    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {model_dir}")

    # key -> file, so we can open only the shard that holds a tensor.
    key_to_file: dict[str, str] = {}
    for f in files:
        with safe_open(f, framework="numpy") as handle:
            for k in handle.keys():
                key_to_file[k] = f

    def get(key: str) -> np.ndarray:
        f = key_to_file.get(key)
        if f is None:
            raise KeyError(f"{key!r} not in any shard of {model_dir}")
        with safe_open(f, framework="numpy") as handle:
            return handle.get_tensor(key).astype(np.float32)

    if num_layers is None:
        num_layers = 0
        while layer_key.format(i=num_layers, proj="gate") in key_to_file:
            num_layers += 1
        if num_layers == 0:
            raise ValueError("could not infer num_layers; pass num_layers=")

    gate, down = [], []
    for i in range(num_layers):
        gate.append(get(layer_key.format(i=i, proj="gate")))
        down.append(get(layer_key.format(i=i, proj="down")))

    embed = get(embed_key)
    try:
        lm_head = get(lm_head_key)
    except KeyError:
        lm_head = embed  # tied
    try:
        final_norm_weight = get(norm_key)
    except KeyError:
        final_norm_weight = None

    return VindexLite(
        gate=gate,
        down=down,
        embed=embed,
        lm_head=lm_head,
        hidden_size=embed.shape[1],
        num_layers=num_layers,
        final_norm_weight=final_norm_weight,
        norm_eps=norm_eps,
        model_name=model_name or model_dir,
        layer_bands=default_layer_bands(num_layers),
    )
