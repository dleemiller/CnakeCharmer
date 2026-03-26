"""
Tests for the training pipeline components.

These tests verify the non-inference parts of the pipeline:
- Prompt formatting
- Tool environment
- Code extraction
- Credit assignment
- Reward computation
"""

import json

import pytest

from cnake_charmer.rewards.composite import composite_reward
from cnake_charmer.training.credit import RolloutNode, mars_credit, mers_credit
from cnake_charmer.training.environment import CythonToolEnvironment
from cnake_charmer.training.prompts import (
    format_feedback,
    format_user_prompt,
    make_initial_messages,
)
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

BAD_CYTHON = """
def primes(nb_primes):
    # This won't compile — invalid cdef in def
    cdef int x = broken syntax here
    return []
"""


# --- Prompt Tests ---


class TestPrompts:
    def test_make_initial_messages(self):
        msgs = make_initial_messages("def foo(): pass", "A test function")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "foo" in msgs[1]["content"]
        assert "A test function" in msgs[1]["content"]

    def test_format_user_prompt(self):
        prompt = format_user_prompt("def add(a, b): return a + b")
        assert "Translate" in prompt
        assert "def add" in prompt

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


# --- Tool Environment ---


class TestToolEnvironment:
    @pytest.fixture()
    def env(self):
        e = CythonToolEnvironment()
        e.reset(
            python_code=PRIMES_PY_CODE,
            func_name="primes",
            test_cases=json.dumps([((10,),), ((20,),)]),
            benchmark_args=json.dumps((100,)),
        )
        return e

    def test_compile_good_code(self, env):
        result = env.compile(PRIMES_CYTHON)
        assert "successful" in result.lower() or "Compilation" in result

    def test_compile_bad_code(self, env):
        result = env.compile(BAD_CYTHON)
        assert "failed" in result.lower() or "error" in result.lower()

    def test_annotate(self, env):
        result = env.annotate(PRIMES_CYTHON)
        assert "score" in result.lower() or "annotation" in result.lower()

    def test_test_correctness(self, env):
        result = env.test(PRIMES_CYTHON)
        assert "2/2" in result or "passed" in result.lower()

    def test_benchmark(self, env):
        result = env.benchmark(PRIMES_CYTHON)
        assert "speedup" in result.lower() or "x" in result.lower()

    def test_composite_score(self, env):
        env.compile(PRIMES_CYTHON)  # sets last_code
        score = env.get_composite_score()
        assert score > 0.3

    def test_tool_methods_exist(self, env):
        assert callable(env.compile)
        assert callable(env.annotate)
        assert callable(env.test)
        assert callable(env.benchmark)
        assert callable(env.reset)


# --- Credit Assignment ---


class TestCredit:
    def test_mars_leaf(self):
        node = RolloutNode(code="x", reward=0.5)
        assert mars_credit(node) == 0.5

    def test_mars_propagation(self):
        root = RolloutNode(
            code="",
            turn=0,
            children=[
                RolloutNode(
                    code="v1",
                    reward=0.3,
                    turn=1,
                    children=[
                        RolloutNode(code="v1.1", reward=0.9, turn=2, solved=True),
                    ],
                ),
                RolloutNode(code="v2", reward=0.7, turn=1, solved=True),
            ],
        )
        mars_credit(root)
        # Root should get credit from best path (0.9 through v1→v1.1)
        assert root.credit > 0.7

    def test_mers_leaf(self):
        node = RolloutNode(code="x", reward=0.5)
        assert mers_credit(node) == 0.5

    def test_mers_smoothing(self):
        root = RolloutNode(
            code="",
            turn=0,
            children=[
                RolloutNode(code="a", reward=0.2, turn=1),
                RolloutNode(code="b", reward=0.8, turn=1),
            ],
        )
        mers_credit(root)
        # MeRS averages, so root credit should be moderate
        assert 0.0 < root.credit < 1.0


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
