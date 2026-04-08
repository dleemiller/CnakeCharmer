"""Wiki page length limits for writer tools."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_WIKI_MAX_TOKENS = 1200


def _count_tokens(text: str) -> tuple[int, str]:
    """Approximate token count with tiktoken fallback to chars/4."""
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.encoding_for_model("gpt-4o-mini")
            return len(enc.encode(text)), "tiktoken:gpt-4o-mini"
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text)), "tiktoken:cl100k_base"
    except Exception:
        return int(round(len(text) / 4.0)), "approx:chars/4"


@dataclass(frozen=True)
class WikiLengthCheck:
    ok: bool
    tokens: int
    max_tokens: int
    method: str
    message: str


def check_wiki_page_length(content: str, max_tokens: int | None = None) -> WikiLengthCheck:
    """Check if wiki page content stays under the token budget."""
    cap = max_tokens
    if cap is None:
        cap = int(os.environ.get("CNAKE_WIKI_MAX_TOKENS", str(DEFAULT_WIKI_MAX_TOKENS)))
    tokens, method = _count_tokens(content)
    if tokens <= cap:
        return WikiLengthCheck(
            ok=True,
            tokens=tokens,
            max_tokens=cap,
            method=method,
            message=f"OK: {tokens} <= {cap} tokens ({method})",
        )
    return WikiLengthCheck(
        ok=False,
        tokens=tokens,
        max_tokens=cap,
        method=method,
        message=(
            f"Wiki page too long: {tokens} > {cap} tokens ({method}). "
            "Split into smaller, focused pages."
        ),
    )
