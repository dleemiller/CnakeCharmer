"""Thomas algorithm solve statistics for generated tridiagonal systems.

Keywords: numerical, tdma, tridiagonal, linear solve, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(0.11, 2.2, 0.13, 1.7, 800, 3))
def tdma_solver_stats(a0: float, b0: float, c0: float, d0: float, n: int, passes: int) -> tuple:
    """Solve generated tridiagonal systems and accumulate residual metrics."""
    checksum = 0.0
    residual_sum = 0.0
    last_x0 = 0.0

    for p in range(passes):
        a = [a0 + 0.0003 * ((i + p) % 11) for i in range(n)]
        b = [b0 + 0.0002 * ((i + 2 * p) % 13) for i in range(n)]
        c = [c0 + 0.00025 * ((i + 3 * p) % 7) for i in range(n)]
        d = [d0 + 0.001 * ((i + 5 * p) % 17) for i in range(n)]

        for i in range(1, n):
            m = a[i - 1] / b[i - 1]
            b[i] -= m * c[i - 1]
            d[i] -= m * d[i - 1]

        x = [0.0] * n
        x[n - 1] = d[n - 1] / b[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

        for i in range(1, n - 1):
            a_i = a0 + 0.0003 * ((i + p) % 11)
            b_i = b0 + 0.0002 * ((i + 2 * p) % 13)
            c_i = c0 + 0.00025 * ((i + 3 * p) % 7)
            d_i = d0 + 0.001 * ((i + 5 * p) % 17)
            r = a_i * x[i - 1] + b_i * x[i] + c_i * x[i + 1] - d_i
            residual_sum += r if r >= 0.0 else -r

        checksum += x[0] + x[n // 2] + x[n - 1]
        last_x0 = x[0]

    return (checksum, residual_sum, last_x0)
