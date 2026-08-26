"""Architecture adapters: map a loaded HF model onto the (gate, up, down,
embed, lm_head, final_norm) tensors MARV operates on.

MARV only needs the gated-MLP triple and the embedding/unembedding pair --
attention, positional encoding, and everything else about a model's
architecture is irrelevant to vindex-style feature extraction. One adapter
covers every model family that shares Llama's module naming (Llama, Mistral,
Qwen2, SmolLM2); shapes come from config.json, nothing is hardcoded per model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FfnLayer:
    """One layer's FFN weights, in the shapes MARV operates on."""

    gate: torch.Tensor  # (intermediate_size, hidden_size) -- one row per feature
    up: torch.Tensor  # (intermediate_size, hidden_size)
    down: torch.Tensor  # (hidden_size, intermediate_size) -- one column per feature


class ArchAdapter:
    """Base interface. One subclass per FFN family."""

    def num_layers(self, model) -> int:
        raise NotImplementedError

    def ffn_layer(self, model, layer_idx: int) -> FfnLayer:
        raise NotImplementedError

    def embed(self, model) -> torch.Tensor:
        """Token embedding matrix, (vocab_size, hidden_size)."""
        raise NotImplementedError

    def lm_head(self, model) -> torch.Tensor:
        """Unembedding matrix, (vocab_size, hidden_size). May be tied to embed()."""
        raise NotImplementedError

    def final_norm(self, model) -> nn.Module:
        raise NotImplementedError


class LlamaStyleFFN(ArchAdapter):
    """Covers any HF model with the standard Llama module naming:
    model.model.layers[i].mlp.{gate,up,down}_proj, model.model.embed_tokens,
    model.lm_head, model.model.norm. This is SmolLM2, Llama, Mistral, and
    Qwen2 -- same names, different shapes read straight from config.
    """

    def num_layers(self, model) -> int:
        return len(model.model.layers)

    def ffn_layer(self, model, layer_idx: int) -> FfnLayer:
        mlp = model.model.layers[layer_idx].mlp
        return FfnLayer(
            gate=mlp.gate_proj.weight.detach(),
            up=mlp.up_proj.weight.detach(),
            down=mlp.down_proj.weight.detach(),
        )

    def embed(self, model) -> torch.Tensor:
        return model.model.embed_tokens.weight.detach()

    def lm_head(self, model) -> torch.Tensor:
        return model.lm_head.weight.detach()

    def final_norm(self, model) -> nn.Module:
        return model.model.norm


def detect_adapter(model) -> ArchAdapter:
    """SmolLM2/Llama/Mistral/Qwen2 all match this shape. Extend here (add an
    ArchAdapter subclass) when a model with different FFN module names or a
    gated-MLP-shaped-differently (e.g. MoE experts, GeGLU) shows up -- don't
    special-case it inside extract.py/probe.py.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        first_mlp = model.model.layers[0].mlp
        if all(hasattr(first_mlp, n) for n in ("gate_proj", "up_proj", "down_proj")):
            return LlamaStyleFFN()
    raise ValueError(
        f"No MARV architecture adapter for {type(model).__name__}; "
        "add one in marv/arch.py"
    )
