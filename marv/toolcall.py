"""Optional domain layer: probes aimed at a model's tool-calling behaviour.

This is a thin wrapper over the generic machinery in marv/context.py and
marv/probe.py -- nothing here is required to use MARV, and nothing here is
tool-call-specific except the prompt scaffolding. Point `DEFAULT_TOOL_PROMPTS`
(or `build_smollm2_tool_prompt`) at whatever behaviour you actually want to
study -- safety, math, factual recall -- or ignore this file entirely and
call `marv.context.describe_prompt` directly.

The contextual-probe primitives (`hidden_states_at_layers`, `describe_prompt`)
used to live here; they moved to marv/context.py and are re-exported below
for back-compat.
"""
from __future__ import annotations

import json

import numpy as np

from .context import describe_prompt, hidden_states_at_layers
from .extract import VindexLite
from .probe import logit_lens, top_features

__all__ = [
    "DEFAULT_TOOL_PROMPTS",
    "SMOLLM2_TOOL_SYSTEM_PROMPT",
    "build_smollm2_tool_prompt",
    "hidden_states_at_layers",
    "describe_prompt",
    "compare_tool_prompt",
    "firing_features_at_layer",
]

# Deliberately generic -- point these at whatever tool-call scaffold the
# model you're testing actually expects (e.g. SmolLM2's <tool_call> tags).
# For SmolLM2 specifically, prefer build_smollm2_tool_prompt() below: these
# bare sentences don't put the model in its documented tool-calling context
# (system prompt + <tools> schema + <tool_call> output format), so they
# under-elicit the behavior you're actually trying to probe.
DEFAULT_TOOL_PROMPTS = [
    "What's the weather in Paris? Use the get_weather function.",
    "Convert 10 miles to kilometers using the unit_converter tool.",
]

# Verbatim from HuggingFaceTB/SmolLM2-1.7B-Instruct's instructions_function_calling.md
# (fetched from the model repo, not guessed) -- the documented system prompt
# SmolLM2 was trained to expect when tools are available.
SMOLLM2_TOOL_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out and refuse to answer.
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<tools>{tools}</tools>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{{"name": "func_name1", "arguments": {{"argument1": "value1", "argument2": "value2"}}}},
... (more tool calls as required)
]</tool_call>"""


def build_smollm2_tool_prompt(tokenizer, tools: list[dict], query: str, chat_template: str | None = None) -> str:
    """Render SmolLM2's real, documented tool-calling context (system prompt
    + JSON tool schemas + query) into the final prompt string, via the
    tokenizer's own chat template -- the actual input shape SmolLM2 was
    fine-tuned to expect, as opposed to DEFAULT_TOOL_PROMPTS's bare sentences.

    `tools` is a list of transformers.utils.get_json_schema(...) dicts.
    Some community fine-tunes ship a tokenizer without `chat_template` set
    even though they share the base checkpoint's vocab; pass the base
    checkpoint's `tokenizer.chat_template` as `chat_template` in that case
    (this sets it on the tokenizer you passed in).
    """
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    system_content = SMOLLM2_TOOL_SYSTEM_PROMPT.format(tools=json.dumps(tools))
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def compare_tool_prompt(
    vindex_base: VindexLite,
    vindex_tuned: VindexLite,
    hidden_base: dict[int, np.ndarray],
    hidden_tuned: dict[int, np.ndarray],
    k: int = 5,
):
    """Logit-lens the same prompt's residual through both checkpoints' unembed
    at matching layers -- did fine-tuning change what the model 'wants to
    say' at the tool-call decision point, and in which layer does that first
    show up.
    """
    results = {}
    for layer in sorted(set(hidden_base) & set(hidden_tuned)):
        idx_b, logits_b = logit_lens(vindex_base, hidden_base[layer], k=k)
        idx_t, logits_t = logit_lens(vindex_tuned, hidden_tuned[layer], k=k)
        results[layer] = {
            "base_top": list(zip(idx_b.tolist(), logits_b.tolist())),
            "tuned_top": list(zip(idx_t.tolist(), logits_t.tolist())),
        }
    return results


def firing_features_at_layer(vindex: VindexLite, hidden: np.ndarray, layer: int, k: int = 10):
    """Which FFN features fire on this residual, at this layer -- feed this
    the same `hidden_states_at_layers` output used above, and cross-reference
    the feature indices against `diff.most_changed()` to see whether
    fine-tuning touched the features actually active at the decision point.
    """
    return top_features(vindex, layer, hidden, k=k)
