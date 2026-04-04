"""
DRY utilities for configuring DSPy language models and prompts.

Shared across trace collection scripts (collect_traces, generate_traces,
sample_openrouter, optimize_prompt) to eliminate ~130 lines of duplication.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "data" / "optimized_prompts"
SEED_PROMPT = PROMPTS_DIR / "seed_prompt.txt"


def configure_dspy_lm(
    model_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    cache: bool = False,
    reasoning_effort: str | None = None,
    **extra_kwargs,
):
    """Build and configure a DSPy LM, auto-detecting remote vs local.

    Returns the configured dspy.LM instance.
    """
    import dspy

    is_remote = model_id.startswith("openrouter/")

    # Resolve API key
    if api_key is None:
        if is_remote:
            api_key = os.environ.get("APIKEY", os.environ.get("OPENROUTER_API_KEY", ""))
            if not api_key:
                raise ValueError("API key required for OpenRouter (set APIKEY env var)")
        else:
            api_key = os.environ.get("APIKEY", "local")

    lm_kwargs = {
        "api_key": api_key,
        "temperature": temperature,
        "cache": cache,
        "max_tokens": max_tokens,
        **extra_kwargs,
    }

    if reasoning_effort:
        if is_remote:
            lm_kwargs.setdefault("extra_body", {})["reasoning_effort"] = reasoning_effort
        else:
            lm_kwargs["reasoning_effort"] = reasoning_effort

    if not is_remote:
        lm_kwargs["api_base"] = base_url or "http://localhost:8000/v1"

    lm = dspy.LM(model_id, **lm_kwargs)
    dspy.settings.configure(lm=lm)
    logger.info(f"Configured LM: {model_id}")
    return lm


def load_optimized_prompt(
    model_id: str | None = None,
    program_path: str | None = None,
    max_iters: int = 5,
) -> tuple[object | None, str]:
    """Load a GEPA-optimized program or seed prompt.

    Args:
        model_id: Model ID to look up optimized prompt by slug.
        program_path: Explicit path to program.json (overrides model_id lookup).
        max_iters: Max iterations for the loaded agent.

    Returns:
        (program, prompt_id) tuple. program is None if using seed/base prompt.
    """
    if program_path:
        path = Path(program_path)
        if not path.exists():
            logger.warning(f"Program not found: {path}")
            return None, "base"
        return _load_program(path, max_iters), path.stem

    if model_id:
        slug = model_slug(model_id)
        path = PROMPTS_DIR / slug / "program.json"
        if path.exists():
            return _load_program(path, max_iters), f"gepa_{slug}"

    if SEED_PROMPT.exists():
        logger.info("Using seed prompt")
        return None, "seed_v1"

    logger.info("Using base signature (no optimized prompt)")
    return None, "base"


def _load_program(path: Path, max_iters: int):
    """Load a GEPA program from path."""
    from scripts.optimize_prompt import CythonReActAgent

    agent = CythonReActAgent(max_iters=max_iters)
    agent.load(path)
    logger.info(f"Loaded optimized program: {path}")
    return agent


def apply_optimized_signatures(react_module, optimized_program, seed_text=None):
    """Apply optimized GEPA signatures or seed prompt to a ReAct module."""
    if optimized_program is not None:
        opt_params = dict(optimized_program.named_parameters())
        for name, param in react_module.named_parameters():
            if name in opt_params:
                opt = opt_params[name]
                if hasattr(opt, "signature") and hasattr(param, "signature"):
                    param.signature = opt.signature
    elif seed_text:
        for _name, param in react_module.named_parameters():
            if hasattr(param, "signature"):
                param.signature = param.signature.with_instructions(seed_text)


def model_slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug."""
    return model_id.replace("/", "_").replace(":", "_")


def get_seed_text() -> str | None:
    """Read the seed prompt text, or None if unavailable."""
    if SEED_PROMPT.exists():
        return SEED_PROMPT.read_text().strip()
    return None
