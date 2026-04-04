"""
Parse Cython HTML annotations to extract optimization metrics.

Cython generates HTML files with color-coded lines indicating:
- Yellow background: Python object interactions (slow)
- White/no background: Pure C operations (fast)
- Lines with + expand to show generated C code

The annotation score measures how well the Cython code avoids
Python-level operations.
"""

import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


@dataclass
class AnnotationResult:
    success: bool
    score: float = 0.0  # 0.0 (all Python) to 1.0 (all C)
    total_lines: int = 0
    yellow_lines: int = 0  # Lines with Python interaction
    white_lines: int = 0  # Pure C lines
    hints: list = field(default_factory=list)
    error: str = ""


class CythonAnnotationParser(HTMLParser):
    """Parse Cython's HTML annotation output to extract line-level metrics."""

    def __init__(self):
        super().__init__()
        self.lines = []  # list of (line_num, score) where score is 0-255 yellow intensity
        self._current_score = None
        self._in_code_line = False
        self._line_num = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        # Cython 3.x generates lines as <pre> or <span> with onclick and style
        # Each code line is wrapped in a tag with a background-color style
        style = attrs_dict.get("style", "")

        # Detect scored lines — Cython uses background color intensity
        # Score 0 = pure C (white), Score 255 = heavy Python (deep yellow)
        if "score" in attrs_dict:
            try:
                self._current_score = int(attrs_dict["score"])
            except (ValueError, TypeError):
                self._current_score = None

        # Alternative: parse background-color from style
        # Cython uses rgb(255, 255, X) where X varies — lower X = more yellow
        bg_match = re.search(r"background-color:\s*#([0-9a-fA-F]{6})", style)
        if bg_match:
            hex_color = bg_match.group(1)
            # Yellow is FFFF00, white is FFFFFF
            # The blue channel indicates intensity: 00 = full yellow, FF = white
            blue = int(hex_color[4:6], 16)
            self._current_score = 255 - blue  # 0 = white/C, 255 = yellow/Python

    def handle_data(self, data):
        pass

    def handle_endtag(self, tag):
        pass


def parse_annotations(
    html_content: str | None = None,
    html_path: str | None = None,
) -> AnnotationResult:
    """
    Parse Cython HTML annotation to extract optimization metrics.

    Provide either html_content (string) or html_path (file path).

    Returns:
        AnnotationResult with score, line counts, and optimization hints.
    """
    if html_content is None and html_path is None:
        return AnnotationResult(success=False, error="No HTML content or path provided")

    if html_content is None:
        try:
            with open(html_path, encoding="utf-8") as f:
                html_content = f.read()
        except Exception as e:
            return AnnotationResult(success=False, error=f"Failed to read HTML: {e}")

    if not html_content.strip():
        return AnnotationResult(success=False, error="Empty HTML content")

    try:
        return _parse_cython_html(html_content)
    except Exception as e:
        logger.error(f"Failed to parse annotation HTML: {e}")
        return AnnotationResult(success=False, error=str(e))


def _parse_cython_html(html: str) -> AnnotationResult:
    """
    Parse Cython annotation HTML using regex patterns.

    Cython 3.x HTML format uses lines with a `score` attribute indicating
    the amount of Python interaction (0 = pure C, higher = more Python).
    """
    result = AnnotationResult(success=True)
    hints = []

    # Pattern 1: Cython 3.x score-based lines
    # <pre class='cython line score-N' ...>
    score_pattern = re.compile(
        r"<pre[^>]*class=['\"]cython line score-(\d+)['\"][^>]*>(.*?)</pre>",
        re.DOTALL,
    )
    matches = list(score_pattern.finditer(html))

    if matches:
        for i, m in enumerate(matches, 1):
            score = int(m.group(1))
            line_text = re.sub(r"<[^>]+>", "", m.group(2)).strip()

            if not line_text:
                continue

            result.total_lines += 1

            if score == 0:
                result.white_lines += 1
            else:
                result.yellow_lines += 1
                # High-score lines get hints
                if score > 50:
                    hints.append(
                        f"Line {i}: high Python interaction (score={score}): {line_text[:80]}"
                    )
    else:
        # Fallback: count lines with background-color highlighting
        # Cython also uses inline styles with background colors
        line_pattern = re.compile(
            r'<(?:div|span|pre)[^>]*style="[^"]*background-color:\s*#([0-9a-fA-F]{6})[^"]*"[^>]*>(.*?)</(?:div|span|pre)>',
            re.DOTALL,
        )
        matches = list(line_pattern.finditer(html))

        if matches:
            for m in matches:
                hex_color = m.group(1).upper()
                line_text = re.sub(r"<[^>]+>", "", m.group(2)).strip()

                if not line_text:
                    continue

                result.total_lines += 1

                # FFFFFF = white (pure C), FFFF00 = yellow (Python interaction)
                if hex_color == "FFFFFF" or hex_color == "":
                    result.white_lines += 1
                else:
                    result.yellow_lines += 1
        else:
            # Last resort: just count non-empty, non-comment code lines
            code_lines = re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL)
            for cl in code_lines:
                text = re.sub(r"<[^>]+>", "", cl).strip()
                if text and not text.startswith("#"):
                    result.total_lines += 1
                    result.white_lines += 1  # Assume C if we can't determine

    # Calculate score
    if result.total_lines > 0:
        result.score = result.white_lines / result.total_lines
    else:
        result.score = 0.0

    # Add general hints based on patterns in the HTML
    if "PyObject" in html:
        py_obj_count = html.count("PyObject")
        if py_obj_count > 10:
            hints.append(
                f"Found {py_obj_count} PyObject references — consider adding more type declarations"
            )

    if "__Pyx_PyObject_Call" in html:
        call_count = html.count("__Pyx_PyObject_Call")
        if call_count > 5:
            hints.append(
                f"Found {call_count} Python function calls — consider using cdef functions"
            )

    if "GIL" in html or "__pyx_gilstate" in html:
        hints.append("GIL acquisition detected — consider using nogil sections")

    result.hints = hints
    return result
