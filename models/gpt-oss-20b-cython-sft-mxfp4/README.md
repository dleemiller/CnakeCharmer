---
base_model: openai/gpt-oss-20b
library_name: transformers
pipeline_tag: text-generation
model_name: CnakeAgent-sft-v0.1
license: apache-2.0
tags:
- cython
- code-generation
- tool-use
- sft
- gpt-oss
- vllm
---

# CnakeAgent-sft-v0.1

> [!NOTE]
> This is a preview checkpoint intended for testing and integration validation.
> Planned RL/GRPO releases will be continued from this SFT checkpoint as the initialization base.

`CnakeAgent-sft-v0.1` is a supervised fine-tune of `openai/gpt-oss-20b` for Python -> Cython optimization workflows.
It is trained on multi-turn tool-use traces where the model proposes Cython code and receives compile/test/benchmark feedback.

This checkpoint is packaged in an MXFP4-compatible format for efficient serving.

## What This Model Is For

- Translating Python functions to optimized Cython
- Iterative refinement with evaluator feedback (`evaluate_cython`)
- Agent-style optimization loops in MCP or OpenAI-compatible tool-calling runtimes

## Recommended Serving (vLLM)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model CnakeCharmer/CnakeAgent-sft-v0.1 \
  --served-model-name gpt-oss-20b-cython \
  --host 0.0.0.0 \
  --port 8003 \
  --trust-remote-code
```

## MCP Usage

> [!IMPORTANT]
> This model is designed primarily as a local agent backend for code tools such as Claude Code and Codex.

> [!NOTE]
> The CnakeCharmer tool-execution path uses Bubblewrap (`bwrap`) for sandboxing.
> Install it before running MCP agent loops that call `evaluate_cython`.
>
> Linux install:
> ```bash
> # Debian / Ubuntu
> sudo apt-get update && sudo apt-get install -y bubblewrap
>
> # Fedora
> sudo dnf install -y bubblewrap
>
> # Arch
> sudo pacman -S --noconfirm bubblewrap
> ```

```bash
# one-time setup
git clone https://github.com/dleemiller/CnakeCharmer.git
cd CnakeCharmer
uv sync

# terminal 1: model server
bash scripts/start_vllm_server.sh

# terminal 2: MCP
uv run python -m cnake_charmer.mcp_server
```

Then call `run_cython_agent` from your MCP client.

## Add MCP To Your Client

### Claude Code

```bash
claude mcp add cnake-charmer -- uv run python -m cnake_charmer.mcp_server
```

### Codex

```bash
codex mcp add cnake-charmer -- uv run python -m cnake_charmer.mcp_server
```

## Typical Workflow

1. Profile your Python application to find hotspots (`cProfile`, `py-spy`, or benchmark timings).
2. Ask your coding agent (Claude Code or Codex) to isolate one target function or tight loop for optimization.
3. Have the coding agent call `run_cython_agent` with the isolated `python_code`, `func_name`, and short task description.
4. Review the returned compile/test/speedup metrics, then apply the generated Cython code into your project.
5. Re-profile and iterate on the next hotspot.

## Direct Inference

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "CnakeCharmer/CnakeAgent-sft-v0.1"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

system_prompt_path = hf_hub_download(model_id, "system_prompt.txt")
with open(system_prompt_path) as f:
    system_prompt = f.read().strip()
user_prompt = (
    "python_code: def add(a, b):\n"
    "    return a + b\n\n"
    "func_name: add\n"
    "description: optimize with cython"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

inputs = tok.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    out = model.generate(inputs, max_new_tokens=512)

print(tok.decode(out[0], skip_special_tokens=True))
```

## Prompting Notes

- The model was trained with a consistent instruction scaffold.
- For best behavior, use server-side default instructions (MCP handles this automatically).
- The checkpoint includes `system_prompt.txt` for reproducible agent behavior.

## Limitations

- Optimized for Cython/tool-use tasks, not general chat.
- Quality depends on evaluator feedback loop quality and test coverage.
- Can still produce non-compiling code in early iterations.

## Training Data

Built from curated tool-use traces in the CnakeCharmer project:
- parallel Python/Cython reference pairs
- multi-turn evaluation traces with compile/test/benchmark feedback

Project repo: https://github.com/dleemiller/CnakeCharmer
