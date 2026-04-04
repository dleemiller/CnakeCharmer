"""
OmegaConf-based configuration management.

Model profiles live in configs/models/ and training configs in configs/training/.
Each profile inherits from _base.yaml and can be overridden via CLI dotlist.

Usage:
    cfg = load_model_profile("gpt_oss_120b")
    cfg = load_model_profile("deepseek_v3", overrides=["model.temperature=0.8"])
    cfg = load_training_config("sft_base")
"""

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_model_profile(name: str, overrides: list[str] | None = None) -> DictConfig:
    """Load a model profile by name (e.g., 'gpt_oss_120b').

    Merges _base.yaml defaults with the named profile, then applies
    any CLI overrides as a dotlist.
    """
    base_path = CONFIGS_DIR / "models" / "_base.yaml"
    profile_path = CONFIGS_DIR / "models" / f"{name}.yaml"

    if not profile_path.exists():
        available = [p.stem for p in (CONFIGS_DIR / "models").glob("*.yaml") if p.stem != "_base"]
        raise FileNotFoundError(
            f"Model profile '{name}' not found. Available: {', '.join(sorted(available))}"
        )

    base = OmegaConf.load(base_path)
    profile = OmegaConf.load(profile_path)
    cfg = OmegaConf.merge(base, profile)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg


def load_training_config(name: str, overrides: list[str] | None = None) -> DictConfig:
    """Load a training config by name (e.g., 'sft_base', 'grpo_base').

    Applies CLI overrides as a dotlist.
    """
    config_path = CONFIGS_DIR / "training" / f"{name}.yaml"

    if not config_path.exists():
        available = [p.stem for p in (CONFIGS_DIR / "training").glob("*.yaml")]
        raise FileNotFoundError(
            f"Training config '{name}' not found. Available: {', '.join(sorted(available))}"
        )

    cfg = OmegaConf.load(config_path)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg


def resolve_api_key(cfg: DictConfig) -> str | None:
    """Resolve API key from config's api_key_env field."""
    env_var = cfg.get("model", {}).get("api_key_env")
    if env_var:
        return os.environ.get(env_var)
    return None


def model_slug(model_id: str) -> str:
    """Convert a model ID to a filesystem-safe slug.

    Example: 'openrouter/deepseek/deepseek-v3.2' → 'openrouter_deepseek_deepseek-v3.2'
    """
    return model_id.replace("/", "_")


def is_remote_model(model_id: str) -> bool:
    """Check if model ID points to a remote API (OpenRouter, etc.)."""
    return model_id.startswith("openrouter/")
