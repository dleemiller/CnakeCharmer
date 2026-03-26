"""
Rollout utilities for Cython code generation training.

With TRL GRPOTrainer + environment_factory, the multi-turn rollout loop
is handled internally by TRL. This module provides helper utilities.
"""

import re


def extract_code_from_content(content: str) -> str:
    """Extract Cython code from model response (may be in a code block).

    Args:
        content: Raw model output text.

    Returns:
        Extracted code string.
    """
    patterns = [
        r"```(?:cython|pyx)\n(.*?)```",
        r"```\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
    return content.strip()
