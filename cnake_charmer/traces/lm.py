"""
DRY utilities for configuring DSPy language models and prompts.

Shared across trace collection scripts (collect_traces, optimize_prompt)
to eliminate duplication of LM setup and prompt loading.
"""

import json
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
    agent = CythonReActAgent(max_iters=max_iters)
    agent.load(path)
    logger.info(f"Loaded optimized program: {path}")
    return agent


class CythonReActAgent:
    """ReAct agent that creates fresh tools per problem.

    GEPA optimizes the instructions in this module's internal
    ReAct predictor across multiple problems. Also used by
    _load_program() to deserialize saved GEPA programs.
    """

    def __init__(self, max_iters: int = 5):
        import dspy

        self.max_iters = max_iters
        self._dspy = dspy
        self._init_react()

    def _init_react(self):
        dspy = self._dspy
        from cnake_charmer.training.dspy_agent import CythonOptimization

        def evaluate_cython(code: str) -> str:
            """Compile, analyze, test, and benchmark Cython code in one step."""
            return "placeholder"

        self.react = dspy.ReAct(
            CythonOptimization,
            tools=[evaluate_cython],
            max_iters=self.max_iters,
        )

        seed_text = get_seed_text()
        if seed_text:
            for _name, param in self.react.named_parameters():
                if hasattr(param, "signature"):
                    param.signature = param.signature.with_instructions(seed_text)

    def forward(
        self,
        python_code: str,
        func_name: str,
        description: str = "",
        test_cases: str = "[]",
        benchmark_args: str = "null",
    ):
        dspy = self._dspy
        from cnake_charmer.training.dspy_agent import CythonOptimization, make_tools

        tc = json.loads(test_cases) if isinstance(test_cases, str) else test_cases
        ba = json.loads(benchmark_args) if isinstance(benchmark_args, str) else benchmark_args

        tools, _env = make_tools(python_code, func_name, tc, ba)
        real_react = dspy.ReAct(CythonOptimization, tools=tools, max_iters=self.max_iters)

        for name, param in self.react.named_parameters():
            for real_name, real_param in real_react.named_parameters():
                if (
                    name == real_name
                    and hasattr(param, "signature")
                    and hasattr(real_param, "signature")
                ):
                    real_param.signature = param.signature

        return real_react(python_code=python_code, func_name=func_name, description=description)

    # Delegate dspy.Module methods for save/load compatibility
    def save(self, path):
        self.react.save(path)

    def load(self, path):
        self.react.load(path)

    def named_parameters(self):
        return self.react.named_parameters()


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
