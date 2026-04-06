"""
ThinkingReAct — ReAct variant that uses native LM thinking mode.

Instead of requiring the LM to emit a [[ ## next_thought ## ]] field,
this module enables the LM's native thinking/reasoning (e.g. Gemma 4
think blocks) and extracts the chain-of-thought from the response
prefix before the structured [[ ## ]] field markers.

Usage:
    react = ThinkingReAct(CythonOptimization, tools=[evaluate_cython])
    pred = react(python_code=code, func_name="foo", description="...")

The module is GEPA-compatible: self.react is a dspy.Predict whose
signature instructions can be optimized.
"""

import logging
import re
import threading
from collections.abc import Callable
from typing import Any, Literal

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import Tool
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)

_FIELD_MARKER_RE = re.compile(r"\[\[ ## \w+ ## \]\]")


def split_thinking(text: str) -> tuple[str, str]:
    """Split native thinking prefix from structured DSPy field output.

    Returns (thinking, structured). ``thinking`` is the model's
    chain-of-thought that appeared *before* the first ``[[ ## ]]`` marker.
    """
    match = _FIELD_MARKER_RE.search(text)
    if match and match.start() > 0:
        thinking = text[: match.start()].strip()
        structured = text[match.start() :]
        # vLLM Gemma 4 decodes <start_of_thought> as the literal word "thought"
        for prefix in ("thought\n", "thought"):
            if thinking.startswith(prefix):
                thinking = thinking[len(prefix) :].strip()
                break
        return thinking, structured
    return "", text


class ThinkingAdapter(ChatAdapter):
    """ChatAdapter that strips native thinking from LM response content.

    Everything before the first ``[[ ## ]]`` marker is treated as the
    model's thinking and stored in thread-local for retrieval by
    :class:`ThinkingReAct`.
    """

    _local = threading.local()

    def parse(self, signature, completion):
        thinking, structured = split_thinking(completion)
        type(self)._local.last_thinking = thinking
        return super().parse(signature, structured)

    @classmethod
    def get_last_thinking(cls) -> str:
        return getattr(cls._local, "last_thinking", "")


class ThinkingReAct(dspy.Module):
    """ReAct agent that uses native LM thinking instead of an explicit thought field.

    Differences from ``dspy.ReAct``:
    - The react signature has **no** ``next_thought`` output field.
    - The model's native thinking (e.g. Gemma 4 ``<think>`` blocks) is
      extracted from the response prefix and stored in the trajectory.
    - A :class:`ThinkingAdapter` is scoped via ``dspy.settings.context``
      during react steps so it doesn't affect other DSPy modules.

    GEPA compatibility: ``self.react`` is a ``dspy.Predict`` with a
    mutable signature, so GEPA can optimise its instructions normally.
    """

    def __init__(
        self,
        signature: type[dspy.Signature] | str,
        tools: list[Callable],
        max_iters: int = 5,
    ):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields])

        instr = [f"{signature.instructions}\n"] if signature.instructions else []
        instr.extend(
            [
                f"You are an Agent. Given {inputs} as input, and your past trajectory,",
                f"use one or more tools to collect information for producing {outputs}.\n",
                "Use your internal reasoning to think through each step.",
                "Then select next_tool_name and provide next_tool_args (as JSON).",
                "After each tool call you receive an observation appended to your trajectory.\n",
                "Available tools:\n",
            ]
        )

        tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=(
                f"Marks the task as complete. Signals that all information "
                f"for producing {outputs} is now available to be extracted."
            ),
            args={},
        )

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append(
            "When providing `next_tool_args`, the value inside the field must be in JSON format"
        )

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append(
                "next_tool_name",
                dspy.OutputField(),
                type_=Literal[tuple(tools.keys())],
            )
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)
        self._adapter = ThinkingAdapter()

    def _format_trajectory(self, trajectory: dict[str, Any]) -> str:
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        sig = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(sig, trajectory)

    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)

        for idx in range(max_iters):
            # Scope ThinkingAdapter to the react predict call only
            with dspy.settings.context(adapter=self._adapter):
                try:
                    pred = self._call_with_truncation(self.react, trajectory, **input_args)
                except ValueError as err:
                    logger.warning(f"Agent failed to select a valid tool: {err}")
                    break

            thinking = ThinkingAdapter.get_last_thinking()
            trajectory[f"thought_{idx}"] = thinking
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](
                    **pred.next_tool_args
                )
            except Exception as err:
                trajectory[f"observation_{idx}"] = (
                    f"Execution error in {pred.next_tool_name}: {err}"
                )

            if pred.next_tool_name == "finish":
                break

        extract = self._call_with_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _call_with_truncation(self, module, trajectory, **input_args):
        from litellm import ContextWindowExceededError

        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded context window, truncating oldest tool call.")
                trajectory = self._truncate_trajectory(trajectory)
        raise ValueError("Context window exceeded after 3 truncation attempts.")

    @staticmethod
    def _truncate_trajectory(trajectory: dict) -> dict:
        keys = list(trajectory.keys())
        if len(keys) < 4:
            raise ValueError("Trajectory too short to truncate (only one tool call).")
        for key in keys[:4]:
            trajectory.pop(key)
        return trajectory
