from __future__ import annotations


def parse_event_headers(data: bytes) -> list[tuple[int, int, int]]:
    """Parse repeated 12-byte records: (type_id, payload_len, event_id)."""
    n = len(data)
    if n % 12 != 0:
        raise ValueError("header buffer length must be a multiple of 12")

    out: list[tuple[int, int, int]] = []
    for i in range(0, n, 12):
        type_id = int.from_bytes(data[i : i + 4], "big", signed=False)
        payload_len = int.from_bytes(data[i + 4 : i + 8], "big", signed=False)
        event_id = int.from_bytes(data[i + 8 : i + 12], "big", signed=False)
        out.append((type_id, payload_len, event_id))
    return out


def filter_headers_by_type(
    headers: list[tuple[int, int, int]], type_id: int
) -> list[tuple[int, int, int]]:
    return [h for h in headers if h[0] == type_id]
