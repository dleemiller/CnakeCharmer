"""Budget-aware ReAct variants.

These wrappers enforce a hard cap on evaluate_cython calls at runtime.
If the cap is reached and the model does not pick `finish` next, execution
stops immediately (no extra tool call, no noisy observation).
"""

from __future__ import annotations

import logging

import dspy

logger = logging.getLogger(__name__)


class BudgetedReAct(dspy.ReAct):
    """dspy.ReAct with hard evaluate_cython budgeting."""

    def __init__(self, *args, max_evaluations: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evaluations = max_evaluations

    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        eval_calls = 0

        for idx in range(max_iters):
            try:
                pred = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, **input_args
                )
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {err}")
                break

            # Hard stop: budget exhausted and model didn't choose finish.
            if (
                self.max_evaluations is not None
                and eval_calls >= self.max_evaluations
                and pred.next_tool_name != "finish"
            ):
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](
                    **pred.next_tool_args
                )
            except Exception as err:
                trajectory[f"observation_{idx}"] = (
                    f"Execution error in {pred.next_tool_name}: {type(err).__name__}: {err}"
                )

            if pred.next_tool_name == "evaluate_cython":
                eval_calls += 1
            if pred.next_tool_name == "finish":
                break

        extract = self._call_with_potential_trajectory_truncation(
            self.extract, trajectory, **input_args
        )
        return dspy.Prediction(trajectory=trajectory, **extract)
