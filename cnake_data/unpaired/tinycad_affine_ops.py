from __future__ import annotations


def apply_affine(
    pts: list[tuple[float, float]],
    m00: float,
    m01: float,
    m10: float,
    m11: float,
    tx: float,
    ty: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in pts:
        out.append((m00 * x + m01 * y + tx, m10 * x + m11 * y + ty))
    return out


def bbox(pts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not pts:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)
