"""Rule-based sentence boundary prediction and annotation helpers."""

from __future__ import annotations


def cpredict(
    token_starts: list[int],
    token_ends: list[int],
    text: str,
    sentence_end_chars: set[str],
    min_sent_len: int = 1,
) -> list[int]:
    """Return indices where sentence breaks should occur."""
    n = len(token_starts)
    if n == 0:
        return []

    boundaries: list[int] = []
    last_boundary = 0

    for i in range(n):
        end = token_ends[i]
        ch = text[end - 1] if end > 0 else ""
        if ch in sentence_end_chars and (i - last_boundary + 1) >= min_sent_len:
            boundaries.append(i)
            last_boundary = i + 1

    if not boundaries or boundaries[-1] != n - 1:
        boundaries.append(n - 1)

    return boundaries


def cset_annotations(
    n_tokens: int,
    boundary_indices: list[int],
) -> list[int]:
    """Encode token-level sentence ids based on boundary indices."""
    sent_ids = [0] * n_tokens
    sent_id = 0
    bpos = 0

    for i in range(n_tokens):
        sent_ids[i] = sent_id
        if bpos < len(boundary_indices) and i == boundary_indices[bpos]:
            sent_id += 1
            bpos += 1

    return sent_ids
