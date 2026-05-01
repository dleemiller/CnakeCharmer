"""Gauss-Seidel and projected Gauss-Seidel updates over CSR matrices."""

from __future__ import annotations


def gauss_seidel(
    ap: list[int],
    aj: list[int],
    ax: list[float],
    x: list[float],
    b: list[float],
    indices: list[int],
) -> None:
    for i in indices:
        start = ap[i]
        end = ap[i + 1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = aj[jj]
            if i == j:
                diag = ax[jj]
            else:
                rsum += ax[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag
        else:
            x[i] = b[i]


def projected_gauss_seidel(
    ap: list[int],
    aj: list[int],
    ax: list[float],
    bp: list[int],
    bj: list[int],
    bx: list[float],
    x: list[float],
    b: list[float],
    c: list[float],
    indices: list[int],
    eps: float = 1e-8,
    omega: float = 1.0,
) -> None:
    for i in indices:
        start = ap[i]
        end = ap[i + 1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = aj[jj]
            if i == j:
                diag = ax[jj]
            else:
                rsum += ax[jj] * x[j]

        if abs(diag + eps) > 1e-16:
            x[i] = (1.0 - omega) * x[i] + omega * (b[i] + eps * x[i] - rsum) / (diag + eps)

        start = bp[i]
        end = bp[i + 1]
        rsum = 0.0
        diag = 0.0
        val = x[i]

        for jj in range(start, end):
            j = bj[jj]
            if i == j:
                diag = bx[jj]
            else:
                rsum += bx[jj] * x[j]

        if diag != 0.0:
            val = (c[i] - rsum) / diag

        x[i] = max(val, x[i])


def symmetric_projected_gauss_seidel(
    ap: list[int],
    aj: list[int],
    ax: list[float],
    bp: list[int],
    bj: list[int],
    bx: list[float],
    x: list[float],
    b: list[float],
    c: list[float],
    indices: list[int],
) -> None:
    projected_gauss_seidel(ap, aj, ax, bp, bj, bx, x, b, c, indices)
    projected_gauss_seidel(ap, aj, ax, bp, bj, bx, x, b, c, list(reversed(indices)))
