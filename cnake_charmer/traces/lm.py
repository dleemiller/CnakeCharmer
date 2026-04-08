"""
DRY utilities for configuring DSPy language models and prompts.

Shared across trace collection scripts (collect_traces, optimize_prompt)
to eliminate duplication of LM setup and prompt loading.
"""

import json
import logging
import os
from pathlib import Path
from typing import Literal

import dspy

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "data" / "optimized_prompts"
SEED_PROMPT = PROMPTS_DIR / "seed_prompt.txt"


def configure_dspy_lm(
    model_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    max_tokens: int = 8192,
    cache: bool = False,
    reasoning_effort: str | None = None,
    **extra_kwargs,
):
    """Build and configure a DSPy LM, auto-detecting remote vs local.

    Returns the configured dspy.LM instance.
    """
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

    if top_p is not None:
        lm_kwargs["top_p"] = top_p

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
    use_thinking: bool = False,
) -> tuple[object | None, str]:
    """Load a GEPA-optimized program or seed prompt.

    Args:
        model_id: Model ID to look up optimized prompt by slug.
        program_path: Explicit path to program.json (overrides model_id lookup).
        max_iters: Max iterations for the loaded agent.
        use_thinking: If True, load as ThinkingReAct agent.

    Returns:
        (program, prompt_id) tuple. program is None if using seed/base prompt.
    """
    if program_path:
        if program_path in {"seed", "seed_v1", "__seed__", "__default__"}:
            if SEED_PROMPT.exists():
                logger.info("Using seed prompt (forced)")
                return None, "seed_v1"
            logger.info("Using base signature (seed prompt forced but not found)")
            return None, "base"
        path = Path(program_path)
        if not path.exists():
            logger.warning(f"Program not found: {path}")
            return None, "base"
        return _load_program(path, max_iters, use_thinking), path.stem

    if model_id:
        slug = model_slug(model_id)
        path = PROMPTS_DIR / slug / "program.json"
        if path.exists():
            return _load_program(path, max_iters, use_thinking), f"gepa_{slug}"

    if SEED_PROMPT.exists():
        logger.info("Using seed prompt")
        return None, "seed_v1"

    logger.info("Using base signature (no optimized prompt)")
    return None, "base"


def _load_program(path: Path, max_iters: int, use_thinking: bool = False):
    """Load a GEPA program from path."""
    agent = CythonReActAgent(max_iters=max_iters, use_thinking=use_thinking)
    agent.load(path)
    logger.info(f"Loaded optimized program: {path}")
    return agent


class CythonReActAgent(dspy.Module):
    """ReAct agent that creates fresh tools per problem.

    GEPA optimizes the instructions in this module's internal
    ReAct predictor across multiple problems. Also used by
    _load_program() to deserialize saved GEPA programs.

    Args:
        max_iters: Maximum evaluate_cython calls per problem.
        use_thinking: If True, use ThinkingReAct (native LM thinking mode)
            instead of dspy.ReAct (explicit next_thought field).
    """

    def __init__(self, max_iters: int = 5, use_thinking: bool = False, include_wiki: bool = False):
        super().__init__()
        self.max_iters = max_iters
        self.use_thinking = use_thinking
        self.include_wiki = include_wiki
        self._init_react()

    @property
    def _react_cls(self):
        if self.use_thinking:
            from cnake_charmer.traces.thinking_react import ThinkingReAct

            return ThinkingReAct
        from cnake_charmer.traces.budgeted_react import BudgetedReAct

        return BudgetedReAct

    def _init_react(self):
        from cnake_charmer.training.dspy_agent import CythonOptimization

        def evaluate_cython(code: str) -> str:
            """Compile, analyze, test, and benchmark Cython code in one step."""
            return "placeholder"

        placeholder_tools = [evaluate_cython]

        if self.include_wiki:
            from cnake_charmer.wiki.search import wiki_page_catalog, wiki_page_catalog_text

            catalog_items = wiki_page_catalog()
            wiki_pages = [it["page"] for it in catalog_items]
            wiki_page_literal = Literal.__getitem__(tuple(wiki_pages)) if wiki_pages else str
            wiki_catalog = wiki_page_catalog_text()
            wiki_read_doc = (
                "Read a full Cython wiki page with concrete fix patterns and examples.\n\n"
                "Prefer wiki_read directly using a page from this startup catalog:\n"
                f"{wiki_catalog}\n\n"
                "Important: `page` must be an exact slug from this list.\n"
                "Usage limit: at most 2 wiki_read calls per attempt."
            )

            def wiki_read(page: str) -> str:
                return "placeholder"

            wiki_read.__annotations__["page"] = wiki_page_literal
            wiki_read.__doc__ = wiki_read_doc
            placeholder_tools.append(wiki_read)

        self.react = self._react_cls(
            CythonOptimization,
            tools=placeholder_tools,
            max_iters=self._effective_max_iters(),
            max_evaluations=self.max_iters,
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
        workflow_mode: str = "HC",
        test_cases: str = "[]",
        benchmark_args: str = "null",
    ):
        from cnake_charmer.training.dspy_agent import CythonOptimization, make_tools

        tc = json.loads(test_cases) if isinstance(test_cases, str) else test_cases
        ba = json.loads(benchmark_args) if isinstance(benchmark_args, str) else benchmark_args

        tools, _env = make_tools(
            python_code,
            func_name,
            tc,
            ba,
            include_wiki=self.include_wiki,
        )
        real_react = self._react_cls(
            CythonOptimization,
            tools=tools,
            max_iters=self._effective_max_iters(),
            max_evaluations=self.max_iters,
        )

        for name, param in self.react.named_parameters():
            for real_name, real_param in real_react.named_parameters():
                if (
                    name == real_name
                    and hasattr(param, "signature")
                    and hasattr(real_param, "signature")
                ):
                    real_param.signature = param.signature

        return real_react(
            python_code=python_code,
            func_name=func_name,
            description=description,
            workflow_mode=workflow_mode,
        )

    def _effective_max_iters(self) -> int:
        # Budget max_iters as evaluate_cython calls; reserve wiki + finish slots.
        return self.max_iters + (2 if self.include_wiki else 0) + 1


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
