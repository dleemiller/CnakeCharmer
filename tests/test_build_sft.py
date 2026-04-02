"""Tests for SFT data preparation pipeline and Harmony format validation."""

import json

import pytest

from cnake_charmer.training.sft_validation import (
    get_analysis_lengths,
    total_analysis_length,
    validate_rendered_example,
    validate_trace_for_rendering,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal valid Harmony text and metadata
# ---------------------------------------------------------------------------

MINIMAL_SYSTEM = (
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\nCurrent date: 2026-04-02\n\n"
    "Reasoning: medium\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message.\n"
    "Calls to these tools must go to the commentary channel: 'functions'.<|end|>"
)

MINIMAL_DEVELOPER = (
    "<|start|>developer<|message|># Instructions\n\n"
    "You are a Cython optimization expert.\n\n"
    "# Tools\n\n## functions\n\nnamespace functions {\n\n"
    "type evaluate_cython = (_: { code: string }) => any;\n\n"
    "} // namespace functions<|end|>"
)

MINIMAL_USER = (
    "<|start|>user<|message|>python_code: def foo(n): return n\n"
    "func_name: foo\n"
    "description: identity function<|end|>"
)

MINIMAL_ANALYSIS = (
    "<|start|>assistant<|channel|>analysis<|message|>I will optimize this function.<|end|>"
)

MINIMAL_TOOL_CALL = (
    "<|start|>assistant to=functions.evaluate_cython<|channel|>commentary json<|message|>"
    '{"code": "def foo(int n): return n"}<|call|>'
)

MINIMAL_TOOL_RESP = (
    "<|start|>functions.evaluate_cython to=assistant<|channel|>commentary<|message|>"
    '"## Compilation\\nSuccess"<|end|>'
)

MINIMAL_FINISH_CALL = (
    "<|start|>assistant to=functions.finish<|channel|>commentary json<|message|>{}<|call|>"
)

MINIMAL_FINISH_RESP = (
    '<|start|>functions.finish to=assistant<|channel|>commentary<|message|>"Completed."<|end|>'
)


def make_valid_text(effort="medium", n_eval_turns=1, include_analysis=True):
    """Build a minimal valid Harmony text for testing."""
    system = MINIMAL_SYSTEM.replace("Reasoning: medium", f"Reasoning: {effort}")
    parts = [system, MINIMAL_DEVELOPER, MINIMAL_USER]
    for _ in range(n_eval_turns):
        if include_analysis:
            parts.append(MINIMAL_ANALYSIS)
        parts.append(MINIMAL_TOOL_CALL)
        parts.append(MINIMAL_TOOL_RESP)
    if include_analysis:
        parts.append(MINIMAL_ANALYSIS)
    parts.append(MINIMAL_FINISH_CALL)
    parts.append(MINIMAL_FINISH_RESP)
    return "".join(parts)


def make_metadata(effort="medium"):
    return {
        "reasoning_effort": effort,
        "problem_id": "test/foo",
        "model": "test-model",
    }


# ---------------------------------------------------------------------------
# A. Format validation tests on rendered output
# ---------------------------------------------------------------------------


class TestPreambleStructure:
    def test_valid_preamble(self):
        text = make_valid_text()
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []

    def test_missing_system(self):
        text = make_valid_text().replace("<|start|>system", "<|start|>user", 1)
        errs = validate_rendered_example(text, make_metadata())
        assert any("preamble" in e.lower() or "system" in e.lower() for e in errs)


class TestSystemMessageContent:
    def test_reasoning_effort_present(self):
        text = make_valid_text(effort="high")
        errs = validate_rendered_example(text, make_metadata(effort="high"))
        assert errs == []

    def test_effort_mismatch(self):
        text = make_valid_text(effort="high")
        errs = validate_rendered_example(text, make_metadata(effort="low"))
        assert any("mismatch" in e.lower() or "effort" in e.lower() for e in errs)


class TestDeveloperMessageContent:
    def test_has_tool_schema(self):
        text = make_valid_text()
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []

    def test_missing_namespace(self):
        text = make_valid_text().replace("namespace functions", "namespace tools")
        errs = validate_rendered_example(text, make_metadata())
        assert any("namespace functions" in e for e in errs)


class TestUserMessageContent:
    def test_has_required_fields(self):
        text = make_valid_text()
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []

    def test_missing_python_code(self):
        text = make_valid_text().replace("python_code:", "source:")
        errs = validate_rendered_example(text, make_metadata())
        assert any("python_code" in e for e in errs)


class TestBodyTurnStructure:
    def test_valid_turns(self):
        text = make_valid_text(n_eval_turns=3)
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []

    def test_analysis_optional(self):
        text = make_valid_text(include_analysis=False)
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []

    def test_missing_tool_response(self):
        # Remove the last tool response
        text = make_valid_text()
        text = text.rsplit("<|start|>functions.finish", 1)[0]
        errs = validate_rendered_example(text, make_metadata())
        assert len(errs) > 0


class TestToolCallJson:
    def test_valid_json(self):
        text = make_valid_text()
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []

    def test_empty_code_rejected(self):
        text = make_valid_text()
        text = text.replace(
            '{"code": "def foo(int n): return n"}',
            '{"code": ""}',
        )
        errs = validate_rendered_example(text, make_metadata())
        assert any("empty" in e.lower() and "code" in e.lower() for e in errs)


class TestNoThinkTags:
    def test_clean_text(self):
        text = make_valid_text()
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []

    def test_leaked_think_tags(self):
        text = make_valid_text()
        text = text.replace(
            "I will optimize this function.",
            "<think>\nI will optimize this function.\n</think>",
        )
        errs = validate_rendered_example(text, make_metadata())
        assert any("think" in e.lower() for e in errs)


class TestTerminalCondition:
    def test_ends_with_tool_response(self):
        text = make_valid_text()
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []

    def test_standalone_assistant_ending(self):
        text = make_valid_text()
        text += "<|start|>assistant<|channel|>final<|message|>Done.<|end|>"
        errs = validate_rendered_example(text, make_metadata())
        assert any("standalone" in e.lower() for e in errs)


class TestSpecialTokenCounts:
    def test_call_response_match(self):
        text = make_valid_text(n_eval_turns=3)
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []

    def test_no_return_token(self):
        text = make_valid_text()
        assert "<|return|>" not in text
        errs = validate_rendered_example(text, make_metadata())
        assert errs == []


# ---------------------------------------------------------------------------
# B. Pipeline logic tests with mock trace data
# ---------------------------------------------------------------------------


class TestTraceValidation:
    def _make_trace(self, **overrides):
        trace = {
            "num_iterations": 2,
            "trajectory": {
                "thought_0": "planning",
                "tool_name_0": "evaluate_cython",
                "tool_args_0": {"code": "def foo(int n): return n"},
                "observation_0": "## Compilation\nSuccess",
                "thought_1": "done",
                "tool_name_1": "finish",
                "tool_args_1": {},
                "observation_1": "Completed.",
            },
        }
        trace.update(overrides)
        return trace

    def test_valid_trace(self):
        errs = validate_trace_for_rendering(self._make_trace())
        assert errs == []

    def test_empty_code_rejected(self):
        trace = self._make_trace()
        trace["trajectory"]["tool_args_0"] = {"code": ""}
        errs = validate_trace_for_rendering(trace)
        assert any("empty" in e.lower() or "code" in e.lower() for e in errs)

    def test_code_as_json_key_rejected(self):
        """The kat-coder bug: code ends up as a JSON key, not value."""
        trace = self._make_trace()
        trace["trajectory"]["tool_args_0"] = {
            "code": "",
            "\n# cython: ...\ndef foo(int n): return n": "annotate",
        }
        errs = validate_trace_for_rendering(trace)
        assert any("code" in e.lower() for e in errs)

    def test_missing_tool_name(self):
        trace = self._make_trace()
        del trace["trajectory"]["tool_name_0"]
        errs = validate_trace_for_rendering(trace)
        assert any("tool_name" in e for e in errs)

    def test_missing_observation_for_eval(self):
        trace = self._make_trace()
        trace["trajectory"]["observation_0"] = ""
        errs = validate_trace_for_rendering(trace)
        assert any("observation" in e for e in errs)

    def test_missing_observation_ok_for_finish(self):
        trace = self._make_trace()
        trace["trajectory"]["observation_1"] = ""
        errs = validate_trace_for_rendering(trace)
        assert errs == []


class TestAnalysisLength:
    def test_total_analysis_length(self):
        text = make_valid_text(n_eval_turns=2)
        lengths = get_analysis_lengths(text)
        # 2 eval turns + 1 finish turn = 3 analysis channels
        assert len(lengths) == 3
        assert all(ln > 0 for ln in lengths)

    def test_total_length_sums(self):
        text = make_valid_text()
        total = total_analysis_length(text)
        assert total == sum(get_analysis_lengths(text))


# ---------------------------------------------------------------------------
# C. Full dataset integration test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/sft_dataset.jsonl").exists(),
    reason="SFT dataset not built yet",
)
class TestFullDatasetValid:
    def test_all_examples_valid(self):
        """Every example in the dataset must pass format validation."""
        failures = []
        with open("data/sft_dataset.jsonl") as f:
            for lineno, line in enumerate(f, 1):
                ex = json.loads(line)
                errs = validate_rendered_example(ex["text"], ex)
                if errs:
                    failures.append((lineno, ex.get("problem_id"), errs[0]))

        if failures:
            msg = f"{len(failures)} examples failed validation:\n"
            for ln, pid, err in failures[:10]:
                msg += f"  L{ln} ({pid}): {err}\n"
            pytest.fail(msg)

    def test_effort_distribution_balanced(self):
        """Effort should be roughly evenly distributed (terciles)."""
        from collections import Counter

        efforts = Counter()
        with open("data/sft_dataset.jsonl") as f:
            for line in f:
                ex = json.loads(line)
                efforts[ex.get("reasoning_effort", "?")] += 1

        total = sum(efforts.values())
        for eff in ("low", "medium", "high"):
            count = efforts.get(eff, 0)
            ratio = count / total
            assert 0.2 < ratio < 0.5, f"Effort '{eff}' is {ratio:.1%} of data (expected ~33%)"

    def test_effort_analysis_lengths_ordered(self):
        """Low effort examples should have shorter analysis than medium, medium shorter than high."""
        by_effort = {"low": [], "medium": [], "high": []}
        with open("data/sft_dataset.jsonl") as f:
            for line in f:
                ex = json.loads(line)
                eff = ex.get("reasoning_effort", "?")
                if eff in by_effort:
                    by_effort[eff].append(total_analysis_length(ex["text"]))

        low_max = max(by_effort["low"])
        med_min = min(by_effort["medium"])
        med_max = max(by_effort["medium"])
        high_min = min(by_effort["high"])

        assert low_max <= med_min + 100, f"low max ({low_max}) overlaps medium min ({med_min})"
        assert med_max <= high_min + 100, f"medium max ({med_max}) overlaps high min ({high_min})"

    def test_no_think_tags_in_dataset(self):
        """No leaked <think> tags anywhere."""
        with open("data/sft_dataset.jsonl") as f:
            for lineno, line in enumerate(f, 1):
                ex = json.loads(line)
                assert "<think>" not in ex["text"], f"L{lineno}: leaked <think> tag"
                assert "</think>" not in ex["text"], f"L{lineno}: leaked </think> tag"
