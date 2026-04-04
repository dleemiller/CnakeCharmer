"""
Tests for the training pipeline components.

These tests verify the non-inference parts of the pipeline:
- Prompt formatting
- Tool environment (3-param: code, python_code, test_code)
- Code extraction
- Credit assignment
- Reward computation
"""

import pytest

from cnake_charmer.eval.pipeline import composite_reward
from cnake_charmer.training.environment import CythonToolEnvironment
from cnake_charmer.training.prompts import format_feedback, format_user_prompt
from cnake_charmer.training.rollout import extract_code_from_content as _extract_code_from_content

# --- Fixtures ---


def primes_py(nb_primes):
    primes_list = []
    n = 2
    while len(primes_list) < nb_primes:
        for prime in primes_list:
            if n % prime == 0:
                break
        else:
            primes_list.append(n)
        n += 1
    return primes_list


PRIMES_PY_CODE = """\
def primes(nb_primes):
    primes_list = []
    n = 2
    while len(primes_list) < nb_primes:
        for prime in primes_list:
            if n % prime == 0:
                break
        else:
            primes_list.append(n)
        n += 1
    return primes_list
"""

PRIMES_CYTHON = """
import cython

def primes(int nb_primes):
    cdef int i
    cdef int p[1000]
    if nb_primes > 1000:
        nb_primes = 1000
    cdef int len_p = 0
    cdef int n = 2
    while len_p < nb_primes:
        for i in p[:len_p]:
            if n % i == 0:
                break
        else:
            p[len_p] = n
            len_p += 1
        n += 1
    return [prime for prime in p[:len_p]]
"""

BAD_CYTHON = """
def primes(nb_primes):
    # This won't compile — invalid cdef in def
    cdef int x = broken syntax here
    return []
"""

PRIMES_TEST_CODE = """\
py.primes(10) == cy.primes(10)
py.primes(20) == cy.primes(20)
"""


# --- Prompt Tests ---


class TestPrompts:
    def test_format_user_prompt_full(self):
        prompt = format_user_prompt("def add(a, b): return a + b", "add", "Add numbers")
        assert "python_code: def add" in prompt
        assert "func_name: add" in prompt
        assert "description: Add numbers" in prompt

    def test_format_user_prompt_code_only(self):
        prompt = format_user_prompt("def add(a, b): return a + b")
        assert prompt == "python_code: def add(a, b): return a + b"
        assert "func_name" not in prompt

    def test_format_feedback_compile_success(self):
        fb = format_feedback("compile", {"success": True, "errors": ""})
        assert "successful" in fb.lower()

    def test_format_feedback_compile_failure(self):
        fb = format_feedback("compile", {"success": False, "errors": "SyntaxError on line 5"})
        assert "SyntaxError" in fb

    def test_format_feedback_test(self):
        fb = format_feedback(
            "test", {"success": True, "passed": 3, "total": 5, "failures": ["Case 2: mismatch"]}
        )
        assert "3/5" in fb
        assert "mismatch" in fb

    def test_format_feedback_annotate(self):
        fb = format_feedback(
            "annotate",
            {
                "success": True,
                "score": 0.85,
                "yellow_lines": 3,
                "total_lines": 20,
                "hints": ["Use cdef"],
            },
        )
        assert "0.85" in fb
        assert "Use cdef" in fb


# --- Code Extraction ---


class TestCodeExtraction:
    def test_extract_from_cython_block(self):
        content = "Here's the code:\n```cython\ndef add(int a, int b):\n    return a + b\n```"
        assert "def add" in _extract_code_from_content(content)

    def test_extract_from_plain_block(self):
        content = "```\ndef add(int a, int b):\n    return a + b\n```"
        assert "def add" in _extract_code_from_content(content)

    def test_extract_raw_code(self):
        content = "def add(int a, int b):\n    return a + b"
        assert "def add" in _extract_code_from_content(content)


# --- Tool Environment (3-param interface) ---


class TestToolEnvironment:
    @pytest.fixture()
    def env(self):
        e = CythonToolEnvironment()
        e.reset()
        return e

    def test_good_code(self, env):
        result = env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert "Compilation" in result
        assert "successful" in result.lower()

    def test_bad_code(self, env):
        result = env.evaluate_cython(BAD_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert "failed" in result.lower() or "error" in result.lower()

    def test_tracks_steps(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert len(env.step_scores) == 1
        assert env.num_tool_calls == 1
        assert env.step_scores[0]["compiled"] is True

    def test_multiple_calls(self, env):
        env.evaluate_cython(BAD_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert len(env.step_scores) == 2
        assert env.num_tool_calls == 2
        assert env.step_scores[0]["compiled"] is False
        assert env.step_scores[1]["compiled"] is True

    def test_correctness_from_test_code(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env.step_scores[0]["correctness"] == 1.0

    def test_failing_tests(self, env):
        # Wrong cython: multiply instead of building prime list
        wrong_cy = "def primes(int n):\n    return [i * 2 for i in range(n)]"
        env.evaluate_cython(wrong_cy, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env.step_scores[0]["correctness"] < 1.0

    def test_annotation_score(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env.step_scores[0]["annotations"] > 0.0

    def test_speedup_measured(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env.step_scores[0]["speedup"] > 1.0

    def test_weighted_total(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env.step_scores[0]["total"] > 0.3

    def test_atomic_reward(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env._get_atomic_reward() > 0.3

    def test_progress_reward_single_call(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env._get_progress_reward() == 0.0

    def test_progress_reward_improvement(self, env):
        env.evaluate_cython(BAD_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env._get_progress_reward() > 0.0

    def test_progress_reward_regression(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        env.evaluate_cython(BAD_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env._get_progress_reward() < 0.0

    def test_bonus_correct_code(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env._get_bonus_reward() >= 0.05

    def test_bonus_failed_code(self, env):
        env.evaluate_cython(BAD_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert env._get_bonus_reward() <= 0.0

    def test_reset_clears_state(self, env):
        env.evaluate_cython(PRIMES_CYTHON, PRIMES_PY_CODE, PRIMES_TEST_CODE)
        assert len(env.step_scores) == 1
        env.reset()
        assert len(env.step_scores) == 0
        assert env.num_tool_calls == 0

    def test_py_cy_module_namespace(self, env):
        """Tests run in a namespace with py and cy modules."""
        py_code = "def add(a, b): return a + b"
        cy_code = "def add(int a, int b): return a + b"
        test_code = "py.add(2, 3) == cy.add(2, 3)\npy.add(0, 0) == cy.add(0, 0)"
        result = env.evaluate_cython(cy_code, py_code, test_code)
        assert "2/2 passed" in result

    def test_setup_lines_in_test_code(self, env):
        """Non-assertion lines (variable assignments) work as setup."""
        py_code = "def add(a, b): return a + b"
        cy_code = "def add(int a, int b): return a + b"
        test_code = "x = 5\ny = 10\npy.add(x, y) == cy.add(x, y)"
        result = env.evaluate_cython(cy_code, py_code, test_code)
        assert "1/1 passed" in result

    def test_tool_methods_for_trl(self, env):
        """Only evaluate_cython should be discoverable as a public method by TRL."""
        public_methods = [
            m
            for m in dir(env)
            if not m.startswith("_") and callable(getattr(env, m)) and m != "reset"
        ]
        assert "evaluate_cython" in public_methods
        assert "compile" not in public_methods
        assert "annotate" not in public_methods
        assert "test" not in public_methods
        assert "benchmark" not in public_methods


# --- Composite Reward ---


class TestCompositeReward:
    def test_good_cython(self):
        scores = composite_reward(
            cython_code=PRIMES_CYTHON,
            python_func=primes_py,
            func_name="primes",
            test_cases=[((10,),), ((20,),)],
            benchmark_args=(100,),
            benchmark_runs=3,
        )
        assert scores["compiled"] is True
        assert scores["correctness"] == 1.0
        assert scores["speedup"] > 1.0
        assert scores["total"] > 0.5

    def test_bad_cython(self):
        scores = composite_reward(
            cython_code=BAD_CYTHON,
            python_func=primes_py,
            func_name="primes",
            test_cases=[((10,),)],
        )
        assert scores["compiled"] is False
        assert scores["total"] == 0.0
