"""
Validation for rendered SFT training examples in Harmony format.

Every SFT example must be a single Harmony-formatted string with this structure:

PREAMBLE (exactly 3 messages):
    <|start|>system<|message|>{auto_system}<|end|>         # Template-generated: identity, reasoning effort, channels
    <|start|>developer<|message|>{instructions+tools}<|end|> # Our system prompt + tool schemas
    <|start|>user<|message|>{problem}<|end|>                # python_code, func_name, description

BODY (1+ repeating turn groups, each is 2-3 messages):
    [OPT] <|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
    <|start|>assistant to=functions.{tool}<|channel|>commentary json<|message|>{json_args}<|call|>
    <|start|>functions.{tool} to=assistant<|channel|>commentary<|message|>{json_resp}<|end|>

TERMINAL: Sequence ends with a tool response, never a standalone assistant message.
"""

import json
import re

# Regex to split rendered text into messages by <|start|>...<|end|> or <|call|>
_MSG_PATTERN = re.compile(r"<\|start\|>(.*?)<\|(?:end|call)\|>", re.DOTALL)

# Extract analysis channel content
_ANALYSIS_PATTERN = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", re.DOTALL)

# Extract tool call JSON
_TOOL_CALL_PATTERN = re.compile(
    r"to=functions\.(\w+)<\|channel\|>commentary json<\|message\|>(.*?)<\|call\|>",
    re.DOTALL,
)

# Extract tool response
_TOOL_RESP_PATTERN = re.compile(
    r"<\|start\|>functions\.(\w+) to=assistant<\|channel\|>commentary<\|message\|>",
)


def _classify_message(msg: str) -> str:
    """Classify a message body (between <|start|> and <|end|>/<|call|>) into a role."""
    if msg.startswith("system"):
        return "system"
    if msg.startswith("developer"):
        return "developer"
    if msg.startswith("user"):
        return "user"
    if msg.startswith("assistant") and "<|channel|>analysis" in msg:
        return "analysis"
    if msg.startswith("assistant to=functions."):
        m = re.search(r"to=functions\.(\w+)", msg)
        return f"call:{m.group(1)}" if m else "call:unknown"
    if msg.startswith("functions."):
        m = re.search(r"functions\.(\w+)", msg)
        return f"resp:{m.group(1)}" if m else "resp:unknown"
    if msg.startswith("assistant"):
        return "assistant_standalone"
    return f"unknown:{msg[:30]}"


def get_analysis_lengths(text: str) -> list[int]:
    """Extract character lengths of all analysis channel content sections."""
    return [len(m.strip()) for m in _ANALYSIS_PATTERN.findall(text)]


def total_analysis_length(text: str) -> int:
    """Total character length of all analysis channels in a rendered example."""
    return sum(get_analysis_lengths(text))


def validate_rendered_example(text: str, metadata: dict) -> list[str]:
    """Validate a rendered Harmony SFT example against the format specification.

    Returns a list of error strings. Empty list means the example is valid.
    """
    errors = []

    # --- Structural parsing ---
    messages = _MSG_PATTERN.findall(text)
    if len(messages) < 4:
        errors.append(
            f"Too few messages: {len(messages)} (need at least 4: system+developer+user+1 turn)"
        )
        return errors

    roles = [_classify_message(m) for m in messages]

    # --- Preamble checks ---
    if roles[:3] != ["system", "developer", "user"]:
        errors.append(f"Bad preamble: expected [system, developer, user], got {roles[:3]}")
        return errors

    system_msg = messages[0]
    developer_msg = messages[1]
    user_msg = messages[2]

    # System message content
    effort_meta = metadata.get("reasoning_effort", "")
    m = re.search(r"Reasoning: (\w+)", system_msg)
    if not m:
        errors.append("System message missing 'Reasoning: {effort}'")
    elif m.group(1) != effort_meta:
        errors.append(f"Effort mismatch: text has '{m.group(1)}', metadata has '{effort_meta}'")

    if "Valid channels" not in system_msg:
        errors.append("System message missing 'Valid channels'")

    # Developer message content
    if "namespace functions" not in developer_msg:
        errors.append("Developer message missing 'namespace functions' (tool schema)")
    if "evaluate_cython" not in developer_msg:
        errors.append("Developer message missing 'evaluate_cython' tool definition")

    # User message content
    if "python_code:" not in user_msg:
        errors.append("User message missing 'python_code:'")
    if "func_name:" not in user_msg:
        errors.append("User message missing 'func_name:'")

    # --- Body structure: repeating (analysis?, call:X, resp:X) ---
    body = roles[3:]
    i = 0
    turn = 0
    while i < len(body):
        # Optional analysis
        if i < len(body) and body[i] == "analysis":
            i += 1

        # Must have call
        if i >= len(body) or not body[i].startswith("call:"):
            errors.append(
                f"Turn {turn}: expected 'call:*' at position {i}, "
                f"got '{body[i] if i < len(body) else 'EOF'}'"
            )
            break
        tool = body[i].split(":")[1]
        i += 1

        # Must have matching response
        if i >= len(body) or body[i] != f"resp:{tool}":
            errors.append(
                f"Turn {turn}: expected 'resp:{tool}' at position {i}, "
                f"got '{body[i] if i < len(body) else 'EOF'}'"
            )
            break
        i += 1
        turn += 1

    if turn == 0 and not errors:
        errors.append("No tool call turns found in body")

    # --- Terminal condition ---
    if roles and roles[-1].startswith("assistant") and not roles[-1].startswith("call:"):
        errors.append(f"Ends with standalone assistant message (role: {roles[-1]})")

    # --- Content checks ---

    # No <think> tags
    if "<think>" in text or "</think>" in text:
        errors.append("Leaked <think>/</ think> tags in rendered text")

    # No <|return|> token
    if "<|return|>" in text:
        errors.append(
            "Contains <|return|> token (should not exist without final assistant message)"
        )

    # <|call|> count matches tool response count
    n_call = text.count("<|call|>")
    n_resp = len(_TOOL_RESP_PATTERN.findall(text))
    if n_call != n_resp:
        errors.append(f"call/response mismatch: {n_call} <|call|> tokens, {n_resp} tool responses")

    # Tool call JSON validity + code check
    for tool_name, json_str in _TOOL_CALL_PATTERN.findall(text):
        try:
            args = json.loads(json_str)
        except json.JSONDecodeError:
            errors.append(f"Invalid JSON in {tool_name} tool call: {json_str[:80]}...")
            continue
        if tool_name == "evaluate_cython" and not args.get("code"):
            errors.append("evaluate_cython call has empty/missing 'code' argument")

    # Analysis channels should have content
    for length in get_analysis_lengths(text):
        if length == 0:
            errors.append("Empty analysis channel found")
            break  # one error is enough

    return errors


def validate_trace_for_rendering(trace: dict) -> list[str]:
    """Validate a raw trace before attempting to build messages from it.

    Returns a list of error strings. Empty list means the trace is safe to render.
    """
    errors = []
    traj = trace.get("trajectory", {})
    n_iters = trace.get("num_iterations", 0)

    if n_iters == 0:
        errors.append("num_iterations is 0")
        return errors

    for i in range(n_iters):
        tool_name = traj.get(f"tool_name_{i}")
        if not tool_name or not isinstance(tool_name, str):
            errors.append(f"Iteration {i}: missing or invalid tool_name")
            continue

        if tool_name == "evaluate_cython":
            tool_args = traj.get(f"tool_args_{i}", {})
            if not isinstance(tool_args, dict):
                errors.append(f"Iteration {i}: tool_args is not a dict")
            elif not tool_args.get("code"):
                errors.append(f"Iteration {i}: evaluate_cython has empty/missing 'code'")

        observation = traj.get(f"observation_{i}", "")
        if (not observation or not isinstance(observation, str)) and tool_name != "finish":
            errors.append(f"Iteration {i}: missing observation for {tool_name}")

    return errors
