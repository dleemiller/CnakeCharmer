from __future__ import annotations


def decode_patch_entries(
    entries: list[tuple[int, int, dict]], is_delta: bool, deletions: list[int]
) -> list[tuple[str, int, int | None, dict | None]]:
    out: list[tuple[str, int, int | None, dict | None]] = []
    for idx, cls, state in entries:
        if cls < 0:
            out.append(("leave", idx, None, {}))
        else:
            out.append(("preserve", idx, cls, state))
    if is_delta:
        for d in deletions:
            out.append(("delete", d, None, None))
    return out
