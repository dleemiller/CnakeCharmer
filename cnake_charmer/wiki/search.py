"""
Reusable wiki search and read functions.

Extracted from mcp_server.py so training code can import them
without depending on the MCP server.
"""

import json
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_WIKI_DIR = _PROJECT_ROOT / "wiki"


def _wiki_pages_dir(wiki_dir: Path | None = None) -> Path:
    return (wiki_dir or _DEFAULT_WIKI_DIR) / "pages"


def _available_pages(wiki_dir: Path | None = None) -> list[str]:
    pages_dir = _wiki_pages_dir(wiki_dir)
    if not pages_dir.exists():
        return []
    return sorted(p.stem for p in pages_dir.glob("*.md"))


def wiki_read(page: str, wiki_dir: Path | None = None) -> str:
    """Read a full wiki page.

    Args:
        page: Page name (e.g. 'memoryviews' or 'memoryviews.md').
        wiki_dir: Optional wiki root directory (default: auto-detect).

    Returns:
        Full markdown content of the page, or JSON error with available pages.
    """
    pages_dir = _wiki_pages_dir(wiki_dir)
    stem = page.removesuffix(".md")
    path = pages_dir / f"{stem}.md"

    if not path.exists():
        available = _available_pages(wiki_dir)
        return json.dumps(
            {"error": f"Page '{page}' not found", "available_pages": available},
            indent=2,
        )

    return path.read_text()


def wiki_search(query: str, max_results: int = 5, wiki_dir: Path | None = None) -> str:
    """Search wiki pages for relevant content.

    Searches page titles and content for query terms. Returns matching
    excerpts sorted by relevance.

    Args:
        query: Search terms (space-separated).
        max_results: Maximum number of results to return.
        wiki_dir: Optional wiki root directory (default: auto-detect).

    Returns:
        JSON array of {page, title, excerpt, score} sorted by relevance.
    """
    pages_dir = _wiki_pages_dir(wiki_dir)
    if not pages_dir.exists():
        return json.dumps({"error": "Wiki not found. Run scaffold first."})

    terms = [t.lower() for t in query.split() if t]
    if not terms:
        return json.dumps({"error": "Empty query"})

    results = []
    for md_path in pages_dir.glob("*.md"):
        content = md_path.read_text()
        content_lower = content.lower()
        page_stem = md_path.stem

        # Score: title match (3x weight) + content term frequency
        score = 0
        for term in terms:
            if term in page_stem:
                score += 3
            score += content_lower.count(term)

        if score == 0:
            continue

        # Extract title from first heading
        title = page_stem.replace("-", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

        # Find best excerpt -- first paragraph containing a query term
        excerpt = ""
        for para in re.split(r"\n\n+", content):
            para_lower = para.lower()
            if any(t in para_lower for t in terms):
                clean = re.sub(r"^#+\s*", "", para).strip()
                excerpt = clean[:200]
                break

        results.append({"page": page_stem, "title": title, "excerpt": excerpt, "score": score})

    results.sort(key=lambda r: r["score"], reverse=True)
    return json.dumps(results[:max_results], indent=2)
