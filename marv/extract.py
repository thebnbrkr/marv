"""vindex extraction: pull gate/down features out of a loaded model into
plain numpy, keyed by (layer, feature_index) -- the unit MARV operates on.

No mmap, no on-disk format, no streaming -- a 135M/1.7B model fits in RAM
whole, so this just walks the loaded torch model and copies tensors to numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arch import ArchAdapter, detect_adapter


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

    def save(self, path: str) -> None:
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
            **{f"gate_{i}": g for i, g in enumerate(self.gate)},
            **{f"down_{i}": d for i, d in enumerate(self.down)},
        )

    @classmethod
    def load(cls, path: str) -> "VindexLite":
        z = np.load(path, allow_pickle=False)
        n = int(z["num_layers"])
        norm_weight = z["final_norm_weight"] if "final_norm_weight" in z else np.zeros(0)
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
    )
