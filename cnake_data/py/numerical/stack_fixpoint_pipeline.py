"""Run fixed-point arithmetic pipeline and summarize results.

Adapted from The Stack v2 Cython candidate:
- blob_id: 459af010814400430dd7b6b4b11abf116ef11993
- filename: fixpoint.pyx

Keywords: numerical, fixed-point, babylonian sqrt, modulo, arithmetic
"""

from cnake_data.benchmarks import python_benchmark

FIX_SHIFT = 16
FIX_ONE = 1 << FIX_SHIFT


def _int2fix(v: int) -> int:
    return v << FIX_SHIFT


def _mul(a: int, b: int) -> int:
    return (a * b) >> FIX_SHIFT


def _div(a: int, b: int) -> int:
    return (a << FIX_SHIFT) // b


def _sqrt_fix(x: int) -> int:
    if x == 0:
        return 0
    xn = _int2fix(1)
    for _ in range(12):
        if xn == 0:
            break
        nxt = (xn + _div(x, xn)) // 2
        if nxt == xn:
            break
        xn = nxt
    return xn


@python_benchmark(args=(12000,))
def stack_fixpoint_pipeline(n: int) -> tuple:
    """Iterate fixed-point transforms and return checksum-style tuple."""
    acc = _int2fix(3)
    step = _int2fix(2)
    root_acc = 0

    for i in range(1, n + 1):
        x = _int2fix((i % 17) + 1)
        acc = _mul(acc + x, step)
        acc = acc - _int2fix(i % 5)
        if acc <= 0:
            acc = _int2fix(1)
        r = _sqrt_fix(acc)
        root_acc = (root_acc + r + i) & 0xFFFFFFFF
        acc = acc % _int2fix(997)

    return (acc, root_acc, acc >> FIX_SHIFT, (root_acc >> 8) & 0xFFFF)
