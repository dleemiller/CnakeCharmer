"""
Tests that GRPO prompts are identical to what the model saw during SFT training.

The SFT pipeline (build_sft.py) renders prompts using:
  - System prompt from data/system_prompt.txt
  - Tool schemas from data/tools.json
  - User format: "python_code: ...\nfunc_name: ...\ndescription: ..."

GRPO must produce byte-identical Harmony-rendered prompts, otherwise the model
encounters out-of-distribution text and tool-calling degrades.
"""

import json
from pathlib import Path

import pytest

SFT_SYSTEM_PROMPT_FILE = Path("data/system_prompt.txt")
SFT_TOOLS_FILE = Path("data/tools.json")

# Skip all tests if data files or tokenizer are missing
pytestmark = pytest.mark.skipif(
    not SFT_SYSTEM_PROMPT_FILE.exists() or not SFT_TOOLS_FILE.exists(),
    reason="SFT data files not found (data/system_prompt.txt, data/tools.json)",
)


@pytest.fixture(scope="module")
def tokenizer():
    """Load the gpt-oss tokenizer (skip if model not available)."""
    model_path = Path("models/gpt-oss-20b-cython-sft-merged")
    if not model_path.exists():
        pytest.skip("Model not available at models/gpt-oss-20b-cython-sft-merged")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path))


@pytest.fixture(scope="module")
def sft_tools():
    return json.loads(SFT_TOOLS_FILE.read_text())


@pytest.fixture(scope="module")
def sft_system_prompt():
    return SFT_SYSTEM_PROMPT_FILE.read_text().strip()


# ---------------------------------------------------------------------------
# Prompt identity tests
# ---------------------------------------------------------------------------


class TestPromptIdentity:
    """Verify GRPO prompts are byte-identical to SFT training data."""

    def _render(self, tokenizer, messages, tools):
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_system_prompt_matches_sft(self, sft_system_prompt):
        """GRPO system prompt loads from the same file SFT used."""
        from cnake_charmer.training.prompts import get_system_prompt

        assert get_system_prompt() == sft_system_prompt

    def test_tools_match_sft(self, sft_tools):
        """GRPO tool schemas load from the same file SFT used."""
        from cnake_charmer.training.prompts import get_tools

        assert get_tools() == sft_tools

    def test_user_format_matches_sft(self):
        """GRPO user prompt uses key-value format matching SFT build_messages()."""
        from cnake_charmer.training.prompts import format_user_prompt

        result = format_user_prompt(
            python_code="def add(a, b): return a + b",
            func_name="add",
            description="Add two numbers",
        )
        assert result == (
            "python_code: def add(a, b): return a + b\nfunc_name: add\ndescription: Add two numbers"
        )

    def test_user_format_omits_empty_fields(self):
        """Empty func_name/description are omitted, not rendered as blank."""
        from cnake_charmer.training.prompts import format_user_prompt

        result = format_user_prompt(python_code="def f(): pass")
        assert result == "python_code: def f(): pass"
        assert "func_name" not in result

    def test_full_prompt_identical(self, tokenizer, sft_tools, sft_system_prompt):
        """Full Harmony-rendered prompt is byte-identical between SFT and GRPO."""
        from cnake_charmer.training.prompts import format_user_prompt, get_system_prompt, get_tools

        code = "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"

        # SFT rendering (exactly how build_sft.py does it)
        sft_messages = [
            {"role": "system", "content": sft_system_prompt},
            {
                "role": "user",
                "content": f"python_code: {code}\nfunc_name: fibonacci\ndescription: Compute nth Fibonacci number",
            },
        ]
        sft_rendered = self._render(tokenizer, sft_messages, sft_tools)

        # GRPO rendering (how build_dataset + train_grpo.py does it)
        grpo_messages = [
            {"role": "system", "content": get_system_prompt()},
            {
                "role": "user",
                "content": format_user_prompt(code, "fibonacci", "Compute nth Fibonacci number"),
            },
        ]
        grpo_rendered = self._render(tokenizer, grpo_messages, get_tools())

        assert sft_rendered == grpo_rendered, (
            f"SFT and GRPO prompts differ!\n"
            f"First difference at char {_find_diff_pos(sft_rendered, grpo_rendered)}"
        )

    def test_full_prompt_identical_multiline_code(self, tokenizer, sft_tools, sft_system_prompt):
        """Identity holds for longer, multi-line code with special characters."""
        from cnake_charmer.training.prompts import format_user_prompt, get_system_prompt, get_tools

        code = (
            "def matrix_multiply(A, B):\n"
            "    rows_A, cols_A = len(A), len(A[0])\n"
            "    rows_B, cols_B = len(B), len(B[0])\n"
            "    result = [[0.0] * cols_B for _ in range(rows_A)]\n"
            "    for i in range(rows_A):\n"
            "        for k in range(cols_A):\n"
            "            for j in range(cols_B):\n"
            "                result[i][j] += A[i][k] * B[k][j]\n"
            "    return result"
        )

        sft_msg = (
            f"python_code: {code}\nfunc_name: matrix_multiply\ndescription: Multiply two matrices"
        )
        sft_rendered = self._render(
            tokenizer,
            [
                {"role": "system", "content": sft_system_prompt},
                {"role": "user", "content": sft_msg},
            ],
            sft_tools,
        )
        grpo_rendered = self._render(
            tokenizer,
            [
                {"role": "system", "content": get_system_prompt()},
                {
                    "role": "user",
                    "content": format_user_prompt(code, "matrix_multiply", "Multiply two matrices"),
                },
            ],
            get_tools(),
        )

        assert sft_rendered == grpo_rendered

    def test_token_ids_identical(self, tokenizer, sft_tools, sft_system_prompt):
        """Token IDs (not just text) are identical — catches tokenizer edge cases."""
        from cnake_charmer.training.prompts import format_user_prompt, get_system_prompt, get_tools

        code = "def add(a, b): return a + b"
        messages = [
            {"role": "system", "content": sft_system_prompt},
            {"role": "user", "content": f"python_code: {code}\nfunc_name: add\ndescription: Add"},
        ]

        sft_ids = tokenizer.apply_chat_template(messages, tools=sft_tools, tokenize=True)
        grpo_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": format_user_prompt(code, "add", "Add")},
            ],
            tools=get_tools(),
            tokenize=True,
        )

        assert sft_ids == grpo_ids, (
            f"Token IDs differ at position {_find_diff_pos(sft_ids, grpo_ids)}"
        )


# ---------------------------------------------------------------------------
# GRPO dataset format tests
# ---------------------------------------------------------------------------


class TestDatasetFormat:
    """Verify build_dataset produces correctly formatted rows."""

    def test_dataset_has_system_message(self):
        """Each prompt starts with system message containing the SFT system prompt."""
        from cnake_charmer.dataset.loader import ProblemSpec
        from cnake_charmer.training.grpo import build_dataset
        from cnake_charmer.training.prompts import get_system_prompt

        problem = ProblemSpec(
            problem_id="test/add",
            description="Add two numbers",
            python_code="def add(a, b): return a + b",
            cython_code="def add(int a, int b): return a + b",
            func_name="add",
            test_cases=[((1, 2),)],
            benchmark_args=(100,),
            category="test",
            difficulty="easy",
            source="test",
            metadata={},
        )
        ds = build_dataset([problem])

        prompt = ds[0]["prompt"]
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[0]["content"] == get_system_prompt()
        assert prompt[1]["role"] == "user"
        assert "python_code:" in prompt[1]["content"]
        assert "func_name: add" in prompt[1]["content"]

    def test_dataset_user_format(self):
        """User message uses key-value format, not markdown."""
        from cnake_charmer.dataset.loader import ProblemSpec
        from cnake_charmer.training.grpo import build_dataset

        problem = ProblemSpec(
            problem_id="test/fib",
            description="Fibonacci",
            python_code="def fib(n): return n",
            cython_code="def fib(int n): return n",
            func_name="fib",
            test_cases=[((5,),)],
            benchmark_args=None,
            category="test",
            difficulty="easy",
            source="test",
            metadata={},
        )
        ds = build_dataset([problem])

        user_content = ds[0]["prompt"][1]["content"]
        # Must NOT contain markdown formatting
        assert "```" not in user_content
        assert "Translate" not in user_content
        # Must contain key-value pairs
        assert user_content.startswith("python_code:")
        assert "func_name: fib" in user_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_diff_pos(a, b):
    """Find position of first difference between two sequences."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))
