"""Customer-facing tests for MCP inference path (run_cython_agent)."""

from __future__ import annotations

import json
import sys
import types

import pytest

import cnake_charmer.mcp_server as mcp_server


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, scripted_items: list):
        self._scripted = list(scripted_items)
        self.posts = []
        self.closed = False

    def post(self, path: str, json: dict):
        self.posts.append((path, json))
        if not self._scripted:
            raise RuntimeError("No scripted response left")
        item = self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)

    def close(self):
        self.closed = True


class _FakeEnv:
    def reset(self):
        return None

    def evaluate_cython(self, code: str, python_code: str, test_code: str) -> str:
        assert isinstance(code, str)
        assert isinstance(python_code, str)
        assert isinstance(test_code, str)
        return (
            "## Compilation\nCompilation successful.\n\n"
            "## Tests\nTests: 2/2 passed\n\n"
            "## Annotation\nAnnotation score: 0.91\n\n"
            "## Benchmark\nSpeedup: 3.20x"
        )


@pytest.fixture
def isolate_prompt_sources(monkeypatch, tmp_path):
    """Make instruction-source behavior deterministic per test."""
    monkeypatch.setattr(mcp_server, "DEFAULT_AGENT_INSTRUCTIONS", "")
    monkeypatch.setattr(mcp_server, "DEFAULT_AGENT_INSTRUCTIONS_FILE", "")
    monkeypatch.setattr(mcp_server, "DEFAULT_AGENT_HF_MODEL_REPO", "")
    monkeypatch.setattr(mcp_server, "DEFAULT_AGENT_HF_MODEL_REVISION", "")
    monkeypatch.setattr(mcp_server, "DEFAULT_AGENT_HF_ALLOW_NETWORK", "0")
    monkeypatch.setattr(mcp_server, "SYSTEM_PROMPT_FILE", tmp_path / "missing_system_prompt.txt")


def test_resolve_instructions_override_wins(isolate_prompt_sources, monkeypatch, tmp_path):
    env_file = tmp_path / "env_prompt.txt"
    env_file.write_text("env file prompt")
    monkeypatch.setattr(mcp_server, "DEFAULT_AGENT_INSTRUCTIONS", "env literal prompt")
    monkeypatch.setattr(mcp_server, "DEFAULT_AGENT_INSTRUCTIONS_FILE", str(env_file))

    text, source = mcp_server._resolve_agent_instructions("override prompt")
    assert text == "override prompt"
    assert source == "tool_override"


def test_resolve_instructions_env_literal(isolate_prompt_sources, monkeypatch):
    monkeypatch.setattr(mcp_server, "DEFAULT_AGENT_INSTRUCTIONS", "env literal prompt")
    text, source = mcp_server._resolve_agent_instructions("")
    assert text == "env literal prompt"
    assert source == "env:CNAKE_AGENT_INSTRUCTIONS"


def test_resolve_instructions_env_file(isolate_prompt_sources, monkeypatch, tmp_path):
    p = tmp_path / "instructions.txt"
    p.write_text("prompt from file")
    monkeypatch.setattr(mcp_server, "DEFAULT_AGENT_INSTRUCTIONS_FILE", str(p))

    text, source = mcp_server._resolve_agent_instructions("")
    assert text == "prompt from file"
    assert source == f"env_file:{p}"


def test_resolve_instructions_hf_cache(monkeypatch, isolate_prompt_sources, tmp_path):
    prompt_path = tmp_path / "hf_prompt.txt"
    prompt_path.write_text("prompt from hf cache")

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kwargs: str(prompt_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(
        mcp_server, "DEFAULT_AGENT_HF_MODEL_REPO", "CnakeCharmer/CnakeAgent-sft-v0.1"
    )

    text, source = mcp_server._resolve_agent_instructions("")
    assert text == "prompt from hf cache"
    assert source.startswith("hf_cache:")


def test_run_cython_agent_react_loop_success(monkeypatch):
    scripted = [
        {
            "status": "in_progress",
            "output": [
                {
                    "type": "function_call",
                    "name": "evaluate_cython",
                    "call_id": "call_1",
                    "arguments": json.dumps(
                        {
                            "code": "def add(int a, int b): return a + b",
                            "python_code": "def add(a, b): return a + b",
                            "test_code": "py.add(1,2) == cy.add(1,2)",
                        }
                    ),
                }
            ],
        },
        {
            "status": "completed",
            "output_text": "Finished optimization",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Finished optimization"}],
                }
            ],
        },
    ]
    fake_client = _FakeClient(scripted)

    monkeypatch.setattr(
        mcp_server, "_resolve_agent_instructions", lambda _: ("sys prompt", "test_src")
    )
    monkeypatch.setattr(mcp_server.httpx, "Client", lambda **kwargs: fake_client)

    import cnake_charmer.training.environment as training_env

    monkeypatch.setattr(training_env, "CythonToolEnvironment", _FakeEnv)

    raw = mcp_server.run_cython_agent(
        python_code="def add(a, b): return a + b",
        func_name="add",
        description="simple add",
        model="gpt-oss-20b-cython",
        base_url="http://localhost:8003/v1",
        max_iters=3,
        reasoning_effort="medium",
    )
    result = json.loads(raw)

    assert result["instructions_source"] == "test_src"
    assert result["instructions_chars"] == len("sys prompt")
    assert result["iterations_run"] == 1
    assert result["status"] == "completed"
    assert result["final_text"] == "Finished optimization"
    assert result["best_metrics"]["tests_passed"] == 2
    assert result["best_metrics"]["tests_total"] == 2
    assert result["best_metrics"]["annotation"] == pytest.approx(0.91)
    assert result["best_metrics"]["speedup"] == pytest.approx(3.2)

    # Ensure the second request included accumulated conversation/tool output.
    assert len(fake_client.posts) == 2
    second_payload = fake_client.posts[1][1]
    assert isinstance(second_payload["input"], list)
    assert second_payload["input"][0]["role"] == "user"
    assert any(item.get("type") == "function_call_output" for item in second_payload["input"][1:])
    assert fake_client.closed is True


def test_run_cython_agent_api_error(monkeypatch):
    fake_client = _FakeClient([RuntimeError("boom")])
    monkeypatch.setattr(
        mcp_server, "_resolve_agent_instructions", lambda _: ("sys prompt", "test_src")
    )
    monkeypatch.setattr(mcp_server.httpx, "Client", lambda **kwargs: fake_client)

    import cnake_charmer.training.environment as training_env

    monkeypatch.setattr(training_env, "CythonToolEnvironment", _FakeEnv)

    raw = mcp_server.run_cython_agent(
        python_code="def add(a, b): return a + b",
        func_name="add",
        model="gpt-oss-20b-cython",
        base_url="http://localhost:8003/v1",
        max_iters=2,
    )
    result = json.loads(raw)
    assert "API error at iteration 0" in result["error"]
    assert result["model"] == "gpt-oss-20b-cython"


def test_run_cython_agent_rejects_invalid_max_iters():
    raw = mcp_server.run_cython_agent(
        python_code="def add(a, b): return a + b",
        func_name="add",
        max_iters=0,
    )
    result = json.loads(raw)
    assert result["error"] == "max_iters must be >= 1"
