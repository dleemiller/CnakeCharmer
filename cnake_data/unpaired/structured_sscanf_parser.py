"""Structured text-row parsing utilities."""

from __future__ import annotations

import re


def split_fixed_fields(line, widths):
    out = []
    start = 0
    for width in widths:
        out.append(line[start : start + width])
        start += width
    return out


def parse_typed_row(line, specs):
    """Parse fields using (name, pattern, cast) specs."""
    values = {}
    cursor = 0
    for name, pattern, cast in specs:
        match = re.match(pattern, line[cursor:])
        if not match:
            raise ValueError(f"cannot parse field {name}")
        token = match.group(0)
        values[name] = cast(token)
        cursor += len(token)
    return values
