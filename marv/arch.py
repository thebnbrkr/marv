"""Architecture adapters: everything MARV needs to know about where things
live inside a loaded HF model, in one place.

Two jobs:

- weights (vindex): the FFN triple and the embedding/unembedding pair,
  read by extract.py and probe.py.
- residual stream (trace, edit, diagnostics): which modules write to the
  residual, which module's input is the per-feature activation, the final
  norm and the unembedding.

The rest of MARV asks the adapter instead of naming modules itself, so a new
architecture is one subclass plus `register_adapter`. Sibling packages do
exactly that (marv-audio registers a Whisper decoder adapter on import).

Adapters refuse models they cannot represent faithfully. Gemma, for one, has
Llama's module names but different maths (GeGLU, scaled embeddings, a
(1 + weight) RMSNorm); accepting it would give subtly wrong numbers
everywhere, so it is rejected rather than guessed at.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FfnLayer:
    """One layer's FFN weights, in the shapes MARV operates on."""

    gate: torch.Tensor  # (intermediate_size, hidden_size) -- one row per feature
    up: torch.Tensor | None  # (intermediate_size, hidden_size); None for an ungated MLP
    down: torch.Tensor  # (hidden_size, intermediate_size) -- one column per feature


class ArchAdapter:
    """Base interface. One subclass per architecture family."""

    # Names of the residual writes each layer makes, in forward order.
    parts: tuple[str, ...] = ("attn", "mlp")

    @classmethod
    def matches(cls, model) -> str | None:
        """None if this adapter can represent `model` exactly, else the reason
        it can't (shown to the user when no adapter matches)."""
        raise NotImplementedError

    # ---- residual stream
    def layers(self, model) -> nn.ModuleList:
        raise NotImplementedError

    def component(self, model, layer: int, part: str) -> nn.Module:
        """The module whose forward OUTPUT is exactly this part's residual write."""
        raise NotImplementedError

    def ffn_in(self, model, layer: int) -> nn.Module:
        """A module whose forward INPUT is exactly the vector the FFN's gate
        rows multiply: the residual after this layer's attention, normalised."""
        raise NotImplementedError

    def ffn_out(self, model, layer: int) -> nn.Linear:
        """The FFN's output projection: its INPUT is the per-feature activation
        vector, and its weight[:, f] is feature f's write direction."""
        raise NotImplementedError

    def final_norm(self, model) -> nn.Module:
        raise NotImplementedError

    def unembed(self, model) -> torch.Tensor:
        """Unembedding matrix, (vocab_size, hidden_size)."""
        raise NotImplementedError

    def inputs(self, tokenizer, prompt, device: str) -> dict:
        """Model kwargs for one forward pass. A dict passes straight through
        (moved to `device`), so callers can hand over audio features or
        anything else a non-text model needs."""
        if isinstance(prompt, dict):
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in prompt.items()}
        return {"input_ids": tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)}

    # ---- weights (vindex)
    def num_layers(self, model) -> int:
        return len(self.layers(model))

    def ffn_layer(self, model, layer_idx: int) -> FfnLayer:
        raise NotImplementedError

    def embed(self, model) -> torch.Tensor:
        """Token embedding matrix, (vocab_size, hidden_size)."""
        raise NotImplementedError

    def lm_head(self, model) -> torch.Tensor:
        """Unembedding matrix, (vocab_size, hidden_size). May be tied to embed()."""
        return self.unembed(model)


class LlamaStyleFFN(ArchAdapter):
    """Llama, Mistral, Qwen2/Qwen3 and SmolLM2: model.model.layers[i] with
    self_attn and a SiLU-gated mlp.{gate,up,down}_proj, model.model.norm (an
    RMSNorm), model.lm_head. Shapes come from config.json.
    """

    MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3")

    @classmethod
    def matches(cls, model) -> str | None:
        inner = getattr(model, "model", None)
        if inner is None or not hasattr(inner, "layers") or not hasattr(model, "lm_head"):
            return "no model.model.layers / model.lm_head"
        mlp = getattr(inner.layers[0], "mlp", None)
        if not all(hasattr(mlp, n) for n in ("gate_proj", "up_proj", "down_proj")):
            return "MLP is not gate_proj/up_proj/down_proj"
        cfg = getattr(model, "config", None)
        model_type = getattr(cfg, "model_type", None)
        if model_type not in cls.MODEL_TYPES:
            return (f"model_type {model_type!r} has Llama-style module names but is not a family "
                    f"MARV has checked ({', '.join(cls.MODEL_TYPES)}); its maths may differ")
        act = getattr(cfg, "hidden_act", "silu")
        if act != "silu":
            return f"hidden_act is {act!r}, not silu"
        return None

    def layers(self, model) -> nn.ModuleList:
        return model.model.layers

    def component(self, model, layer: int, part: str) -> nn.Module:
        blk = model.model.layers[layer]
        if part == "attn":
            return blk.self_attn
        if part == "mlp":
            return blk.mlp
        raise ValueError(f"part must be one of {self.parts}, got {part!r}")

    def ffn_in(self, model, layer: int) -> nn.Module:
        return model.model.layers[layer].mlp

    def ffn_out(self, model, layer: int) -> nn.Linear:
        return model.model.layers[layer].mlp.down_proj

    def final_norm(self, model) -> nn.Module:
        return model.model.norm

    def unembed(self, model) -> torch.Tensor:
        return model.lm_head.weight.detach()

    def ffn_layer(self, model, layer_idx: int) -> FfnLayer:
        mlp = model.model.layers[layer_idx].mlp
        return FfnLayer(
            gate=mlp.gate_proj.weight.detach(),
            up=mlp.up_proj.weight.detach(),
            down=mlp.down_proj.weight.detach(),
        )

    def embed(self, model) -> torch.Tensor:
        return model.model.embed_tokens.weight.detach()


_ADAPTERS: list[type[ArchAdapter]] = [LlamaStyleFFN]


def register_adapter(cls: type[ArchAdapter]) -> type[ArchAdapter]:
    """Make detect_adapter consider `cls` (checked before the built-ins).
    Usable as a class decorator."""
    if cls not in _ADAPTERS:
        _ADAPTERS.insert(0, cls)
    return cls


def detect_adapter(model) -> ArchAdapter:
    """The first registered adapter that can represent `model` exactly.
    Raises, listing each adapter's reason, when none can."""
    reasons = []
    for cls in _ADAPTERS:
        why = cls.matches(model)
        if why is None:
            return cls()
        reasons.append(f"{cls.__name__}: {why}")
    raise ValueError(
        f"No MARV architecture adapter for {type(model).__name__}:\n  " + "\n  ".join(reasons)
        + "\nAdd one in marv/arch.py (or register_adapter from your own package)."
    )
