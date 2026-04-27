"""
MCP server exposing CnakeCharmer validation tools.

These are the same tools used during GRPO training — compile, annotate,
test, benchmark, and composite_reward. Claude Code can call them to
iterate on Cython implementations while adding new problems.

All code execution is sandboxed via bubblewrap (bwrap) with filesystem
and network isolation, resource limits, and wall-clock watchdog.

Usage:
    claude mcp add cnake-charmer -- uv run python -m cnake_charmer.mcp_server
"""

import ast
import json
import logging
import os
import re
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from cnake_charmer.eval.annotations import parse_annotations
from cnake_charmer.eval.compiler import cleanup_build, compile_cython
from cnake_charmer.eval.memory_safety import check_memory_safety
from cnake_charmer.eval.pipeline import composite_reward as _composite_reward
from cnake_data.loader import discover_pairs

logger = logging.getLogger(__name__)

mcp = FastMCP("cnake-charmer")

CNAKE_DATA_ROOT = Path(__file__).resolve().parent.parent / "cnake_data"
PY_DIR = CNAKE_DATA_ROOT / "py"
CY_DIR = CNAKE_DATA_ROOT / "cy"
SYSTEM_PROMPT_FILE = Path(__file__).resolve().parent.parent / "data" / "system_prompt.txt"

DEFAULT_AGENT_BASE_URL = os.environ.get("CNAKE_AGENT_BASE_URL", "http://localhost:8003/v1")
DEFAULT_AGENT_MODEL = os.environ.get("CNAKE_AGENT_MODEL", "gpt-oss-20b-cython")
DEFAULT_AGENT_INSTRUCTIONS = os.environ.get("CNAKE_AGENT_INSTRUCTIONS", "")
DEFAULT_AGENT_INSTRUCTIONS_FILE = os.environ.get("CNAKE_AGENT_INSTRUCTIONS_FILE", "")
DEFAULT_AGENT_HF_MODEL_REPO = os.environ.get(
    "CNAKE_AGENT_HF_MODEL_REPO", "CnakeCharmer/CnakeC-sft-v0.1"
)
DEFAULT_AGENT_HF_MODEL_REVISION = os.environ.get("CNAKE_AGENT_HF_MODEL_REVISION", "")
DEFAULT_AGENT_HF_ALLOW_NETWORK = os.environ.get("CNAKE_AGENT_HF_ALLOW_NETWORK", "0")

# Responses API tool schema used by test harness and agent loop.
RESPONSE_TOOLS = [
    {
        "type": "function",
        "name": "evaluate_cython",
        "description": (
            "Compile Cython code, test equivalence against Python reference, and benchmark. "
            "The test_code runs in a namespace where `py` is the Python module and `cy` is "
            "the compiled Cython module. Each test assertion has a 5-second timeout."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Complete .pyx Cython source code."},
                "python_code": {
                    "type": "string",
                    "description": "Original Python source code (reference implementation).",
                },
                "test_code": {
                    "type": "string",
                    "description": "Equivalence test assertions comparing py.<name>(...) == cy.<name>(...).",
                },
            },
            "required": ["code", "python_code", "test_code"],
        },
    }
]


# ---------------------------------------------------------------------------
# Primary tool: score a problem by name (reads files from repo)
# ---------------------------------------------------------------------------


@mcp.tool()
def score_problem(problem_id: str) -> str:
    """Score a Python/Cython problem pair by its ID (e.g. 'numerical/great_circle').

    Reads the .py and .pyx files from the repo, extracts test cases from
    the test file, and runs the full composite reward: compilation,
    correctness, speedup, and annotation quality.

    Args:
        problem_id: Problem path like 'numerical/great_circle' or 'algorithms/primes'.

    Returns:
        JSON with compiled, correctness, speedup, annotation score, total reward,
        and actionable hints for improvement.
    """
    pairs = discover_pairs()
    match = None
    for p in pairs:
        if p.problem_id == problem_id:
            match = p
            break

    if match is None:
        available = sorted(p.problem_id for p in pairs)
        return json.dumps(
            {
                "error": f"Problem '{problem_id}' not found",
                "available": available,
            },
            indent=2,
        )

    if not match.has_cython:
        return json.dumps({"error": f"No Cython implementation found for '{problem_id}'"})

    # Use python_code string for sandboxed execution (no in-process exec)
    scores = _composite_reward(
        cython_code=match.cython_code,
        python_code=match.python_code,
        func_name=match.func_name,
        test_cases=match.test_cases,
        benchmark_args=match.benchmark_args,
        benchmark_runs=3,
    )

    return json.dumps(
        {
            "problem_id": problem_id,
            "func_name": match.func_name,
            "compiled": scores["compiled"],
            "correctness": scores["correctness"],
            "speedup": round(scores["speedup"], 2),
            "annotation_score": round(scores["annotations"], 3),
            "lint_score": round(scores.get("lint", 0.0), 3),
            "memory_safety_score": round(scores.get("memory_safety", 1.0), 3),
            "total_reward": round(scores["total"], 3),
            "annotation_hints": scores["annotation_hints"],
            "lint_violations": scores.get("lint_violations", []),
            "memory_safety_errors": scores.get("memory_safety_errors", []),
            "correctness_failures": scores["correctness_failures"],
            "compilation_errors": scores["compilation_errors"],
        },
        indent=2,
    )


@mcp.tool()
def list_problems() -> str:
    """List all problem pairs in the dataset with their status.

    Returns:
        JSON array of problems with id, func_name, has_cython, category, and test count.
    """
    pairs = discover_pairs()
    result = []
    for p in sorted(pairs, key=lambda x: x.problem_id):
        result.append(
            {
                "problem_id": p.problem_id,
                "func_name": p.func_name,
                "has_cython": p.has_cython,
                "category": p.category,
                "num_tests": len(p.test_cases),
                "benchmark_args": list(p.benchmark_args) if p.benchmark_args else None,
            }
        )
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# evaluate_cython: same interface as the training tool
# ---------------------------------------------------------------------------


@mcp.tool()
def evaluate_cython(code: str, python_code: str, test_code: str) -> str:
    """Compile Cython code, test equivalence against Python reference, and benchmark.

    The test_code runs in a namespace where `py` is the Python module and
    `cy` is the compiled Cython module. Each test assertion has a 5-second timeout.

    Args:
        code: Complete .pyx Cython source code.
        python_code: Original Python source code (reference implementation).
        test_code: Equivalence test assertions comparing py.<name>(...) == cy.<name>(...).

    Returns:
        Compilation status, annotation score, test results, and benchmark speedup.
    """
    from cnake_charmer.training.environment import CythonToolEnvironment

    env = CythonToolEnvironment()
    env.reset()
    return env.evaluate_cython(code=code, python_code=python_code, test_code=test_code)


# ---------------------------------------------------------------------------
# File-based tools (for iterating on implementations)
# ---------------------------------------------------------------------------

SIMD_FLAGS = ["-mavx2", "-mfma", "-O3"]


def _detect_compile_flags(pyx_path: str) -> list:
    """Auto-detect compiler flags from file path.

    Files in cy_simd/ or nn_ops/ get SIMD flags automatically.
    """
    if "cy_simd" in pyx_path or "nn_ops" in pyx_path:
        return SIMD_FLAGS
    return []


@mcp.tool()
def compile_file(pyx_path: str) -> str:
    """Compile a .pyx file and check for errors.

    Auto-detects SIMD flags for cy_simd/ and nn_ops/ files.

    Args:
        pyx_path: Path to a .pyx file, e.g. 'cnake_charmer/cy_simd/nn_ops/relu.pyx'.

    Returns:
        JSON with success status and any error messages.
    """
    path = Path(pyx_path)
    if not path.exists():
        return json.dumps({"success": False, "errors": f"File not found: {pyx_path}"})

    flags = _detect_compile_flags(pyx_path)
    code = path.read_text()
    result = compile_cython(code, annotate=False, extra_compile_args=flags)
    output = {
        "success": result.success,
        "errors": result.errors,
        "warnings": result.warnings,
        "flags": flags,
    }
    cleanup_build(result)
    return json.dumps(output, indent=2)


@mcp.tool()
def annotate_file(pyx_path: str) -> str:
    """Compile a .pyx file and analyze HTML annotations for optimization quality.

    Auto-detects SIMD flags for cy_simd/ and nn_ops/ files.

    Args:
        pyx_path: Path to a .pyx file, e.g. 'cnake_charmer/cy/numerical/great_circle.pyx'.

    Returns:
        JSON with score, yellow/white line counts, and optimization hints.
    """
    path = Path(pyx_path)
    if not path.exists():
        return json.dumps({"success": False, "errors": f"File not found: {pyx_path}"})

    flags = _detect_compile_flags(pyx_path)
    code = path.read_text()
    result = compile_cython(code, annotate=True, keep_build=True, extra_compile_args=flags)

    if not result.success:
        output = {"success": False, "errors": result.errors, "score": 0.0, "hints": []}
        cleanup_build(result)
        return json.dumps(output, indent=2)

    ann = parse_annotations(html_path=result.html_path) if result.html_path else None
    cleanup_build(result)

    if ann and ann.success:
        return json.dumps(
            {
                "success": True,
                "score": round(ann.score, 3),
                "total_lines": ann.total_lines,
                "white_lines": ann.white_lines,
                "yellow_lines": ann.yellow_lines,
                "hints": ann.hints,
            },
            indent=2,
        )

    return json.dumps({"success": True, "score": 0.0, "hints": ["Could not parse annotations"]})


@mcp.tool()
def check_memory(pyx_path: str, func_name: str, test_args: str = "(100,)") -> str:
    """Run AddressSanitizer on a .pyx file to detect memory errors.

    Compiles with -fsanitize=address and runs the function with small inputs.
    Detects leaks, buffer overflows, use-after-free, and double-free.

    Args:
        pyx_path: Path to a .pyx file.
        func_name: Name of the function to test.
        test_args: Python tuple literal for test arguments, e.g. '(100,)'.

    Returns:
        JSON with score (1.0 = clean, 0.0 = errors), error details, and leak bytes.
    """
    path = Path(pyx_path)
    if not path.exists():
        return json.dumps({"success": False, "errors": [f"File not found: {pyx_path}"]})

    try:
        args = ast.literal_eval(test_args)
        if not isinstance(args, tuple):
            args = (args,)
    except (ValueError, SyntaxError) as e:
        return json.dumps({"success": False, "errors": [f"Invalid test_args: {e}"]})

    code = path.read_text()
    flags = _detect_compile_flags(pyx_path)
    result = check_memory_safety(
        cython_code=code,
        func_name=func_name,
        test_args=args,
        extra_compile_args=flags,
    )

    return json.dumps(
        {
            "success": result.success,
            "score": result.score,
            "error_count": result.error_count,
            "leak_bytes": result.leak_bytes,
            "error_types": result.error_types,
            "errors": result.errors,
        },
        indent=2,
    )


def _parse_metrics(result: str) -> dict:
    """Extract compile/test/annotation/speedup metrics from evaluate_cython output text."""
    metrics = {
        "compiled": "Compilation successful" in result,
        "tests_passed": 0,
        "tests_total": 0,
        "annotation": 0.0,
        "speedup": 0.0,
    }
    for line in result.split("\n"):
        m = re.search(r"Tests: (\d+)/(\d+) passed", line)
        if m:
            metrics["tests_passed"] = int(m.group(1))
            metrics["tests_total"] = int(m.group(2))
        m = re.search(r"Annotation score: ([\d.]+)", line)
        if m:
            metrics["annotation"] = float(m.group(1))
        m = re.search(r"Speedup: ([\d.]+)x", line)
        if m:
            metrics["speedup"] = float(m.group(1))
    return metrics


def _resolve_agent_instructions(instructions_override: str = "") -> tuple[str, str]:
    """Resolve MCP agent instructions with explicit precedence.

    Precedence:
      1) tool arg override (`instructions_override`)
      2) env literal (`CNAKE_AGENT_INSTRUCTIONS`)
      3) env file path (`CNAKE_AGENT_INSTRUCTIONS_FILE`)
      4) HF model cache (`CNAKE_AGENT_HF_MODEL_REPO` + system_prompt.txt)
      5) repo default (`data/system_prompt.txt`)
      6) empty string
    """
    if instructions_override and instructions_override.strip():
        return instructions_override.strip(), "tool_override"

    if DEFAULT_AGENT_INSTRUCTIONS and DEFAULT_AGENT_INSTRUCTIONS.strip():
        return DEFAULT_AGENT_INSTRUCTIONS.strip(), "env:CNAKE_AGENT_INSTRUCTIONS"

    if DEFAULT_AGENT_INSTRUCTIONS_FILE and Path(DEFAULT_AGENT_INSTRUCTIONS_FILE).exists():
        return (
            Path(DEFAULT_AGENT_INSTRUCTIONS_FILE).read_text().strip(),
            f"env_file:{DEFAULT_AGENT_INSTRUCTIONS_FILE}",
        )

    # Prefer model-adjacent prompt in Hugging Face cache by explicit repo id.
    if DEFAULT_AGENT_HF_MODEL_REPO:
        try:
            from huggingface_hub import hf_hub_download

            local_only = str(DEFAULT_AGENT_HF_ALLOW_NETWORK).strip() not in {"1", "true", "TRUE"}
            kwargs = {
                "repo_id": DEFAULT_AGENT_HF_MODEL_REPO,
                "repo_type": "model",
                "filename": "system_prompt.txt",
                "local_files_only": local_only,
            }
            if DEFAULT_AGENT_HF_MODEL_REVISION:
                kwargs["revision"] = DEFAULT_AGENT_HF_MODEL_REVISION
            prompt_path = Path(hf_hub_download(**kwargs))
            if prompt_path.exists():
                return prompt_path.read_text().strip(), f"hf_cache:{prompt_path}"
        except Exception:
            # Silent fallback to repo-local prompt if HF cache isn't available.
            pass

    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text().strip(), f"repo_file:{SYSTEM_PROMPT_FILE}"

    return "", "empty"


@mcp.tool()
def run_cython_agent(
    python_code: str,
    func_name: str,
    description: str = "",
    model: str = DEFAULT_AGENT_MODEL,
    base_url: str = DEFAULT_AGENT_BASE_URL,
    max_iters: int = 5,
    reasoning_effort: str = "medium",
    instructions_override: str = "",
) -> str:
    """Run the full Cython agent (model + tool loop) via Responses API.

    This puts both the LLM agent and the evaluate_cython tooling behind MCP,
    so MCP clients can submit Python snippets and receive agent outputs.

    Args:
        python_code: Source Python code to translate/optimize.
        func_name: Name of the target function in python_code.
        description: Optional task description/keywords.
        model: OpenAI-compatible model id served at base_url.
        base_url: OpenAI-compatible API base URL (expects /v1/responses).
        max_iters: Maximum tool-call iterations to run.
        reasoning_effort: low | medium | high.
        instructions_override: Optional explicit instructions text. If empty,
            MCP resolves from env/file defaults.

    Returns:
        JSON object with final response text, tool-call history, and best eval metrics.
    """
    from cnake_charmer.training.environment import CythonToolEnvironment

    if max_iters < 1:
        return json.dumps({"error": "max_iters must be >= 1"}, indent=2)

    system_prompt, prompt_source = _resolve_agent_instructions(instructions_override)
    user_content = (
        f"python_code: {python_code}\n\nfunc_name: {func_name}\ndescription: {description}"
    )

    env = CythonToolEnvironment()
    env.reset()

    request = {
        "model": model,
        "instructions": system_prompt,
        "input": user_content,
        "tools": RESPONSE_TOOLS,
        "max_output_tokens": 8192,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}

    conversation = []
    history = []
    best_metrics = None
    final_text = ""
    final_status = "unknown"

    try:
        client = httpx.Client(base_url=base_url, timeout=180)
    except Exception as e:
        return json.dumps({"error": f"Failed to create HTTP client: {e}"}, indent=2)

    try:
        for iteration in range(max_iters):
            try:
                resp = client.post("/responses", json=request)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                return json.dumps(
                    {
                        "error": f"API error at iteration {iteration}: {e}",
                        "base_url": base_url,
                        "model": model,
                    },
                    indent=2,
                )

            final_status = result.get("status", "unknown")
            if result.get("output_text"):
                final_text = result["output_text"]

            tool_call = None
            for item in result.get("output", []):
                if item.get("type") == "function_call" and item.get("name") == "evaluate_cython":
                    tool_call = item
                    break

            # No tool call => assistant likely produced final answer.
            if not tool_call:
                break

            try:
                args = json.loads(tool_call.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            code = args.get("code", "")
            py = args.get("python_code", python_code)
            test_code = args.get("test_code", "")
            eval_result = env.evaluate_cython(code=code, python_code=py, test_code=test_code)
            metrics = _parse_metrics(eval_result)

            if best_metrics is None or (
                metrics["tests_passed"] > best_metrics["tests_passed"]
                or (
                    metrics["tests_passed"] == best_metrics["tests_passed"]
                    and metrics["speedup"] > best_metrics["speedup"]
                )
            ):
                best_metrics = metrics

            history.append(
                {
                    "iteration": iteration,
                    "tool_name": "evaluate_cython",
                    "metrics": metrics,
                }
            )

            for item in result.get("output", []):
                conversation.append(item)
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.get("call_id", ""),
                    "output": eval_result,
                }
            )

            request["input"] = [
                {"type": "message", "role": "user", "content": user_content},
                *conversation,
            ]

    finally:
        client.close()

    return json.dumps(
        {
            "base_url": base_url,
            "model": model,
            "instructions_source": prompt_source,
            "instructions_chars": len(system_prompt),
            "status": final_status,
            "iterations_run": len(history),
            "best_metrics": best_metrics or {},
            "history": history,
            "final_text": final_text,
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
