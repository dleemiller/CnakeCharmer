"""Minimal wiki read + catalog helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_WIKI_DIR = _PROJECT_ROOT / "wiki"
_INDEX_ENTRY_RE = re.compile(r"^- \[([^\]]+)\]\(pages/([^)]+)\)\s*(?:\u2014|-)\s*(.+)$")


def _wiki_pages_dir(wiki_dir: Path | None = None) -> Path:
    return (wiki_dir or _DEFAULT_WIKI_DIR) / "pages"


def _available_pages(wiki_dir: Path | None = None) -> list[str]:
    pages_dir = _wiki_pages_dir(wiki_dir)
    if not pages_dir.exists():
        return []
    return sorted(p.stem for p in pages_dir.glob("*.md"))


def _build_catalog(pages_dir: Path) -> list[tuple[str, str]]:
    wiki_dir = pages_dir.parent
    index_path = wiki_dir / "index.md"

    items: list[tuple[str, str]] = []
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            m = _INDEX_ENTRY_RE.match(line.strip())
            if not m:
                continue
            title, rel_path, desc = m.groups()
            page = Path(rel_path).stem
            if not (pages_dir / f"{page}.md").exists():
                continue
            summary = f"{title}: {desc.strip()}"
            items.append((page, summary))

    seen = {p for p, _ in items}
    for page in sorted(p.stem for p in pages_dir.glob("*.md")):
        if page in seen:
            continue
        items.append((page, page.replace("-", " ")))

    return items


def wiki_page_catalog(wiki_dir: Path | None = None) -> list[dict[str, str]]:
    """Return available wiki pages with short summaries from wiki/index.md."""
    pages_dir = _wiki_pages_dir(wiki_dir)
    if not pages_dir.exists():
        return []
    return [{"page": page, "summary": summary} for page, summary in _build_catalog(pages_dir)]


def wiki_page_catalog_text(wiki_dir: Path | None = None) -> str:
    """Formatted wiki page catalog for tool descriptions/prompts."""
    catalog = wiki_page_catalog(wiki_dir)
    if not catalog:
        return "- (no wiki pages found)"
    return "\n".join(f"- {it['page']}: {it['summary']}" for it in catalog)


def wiki_read(page: str, wiki_dir: Path | None = None) -> str:
    """Read a full wiki page by exact page slug."""
    pages_dir = _wiki_pages_dir(wiki_dir)
    stem = page.removesuffix(".md")
    path = pages_dir / f"{stem}.md"
    if not path.exists():
        return json.dumps(
            {"error": f"Page '{page}' not found", "available_pages": _available_pages(wiki_dir)},
            indent=2,
        )
    return path.read_text()
