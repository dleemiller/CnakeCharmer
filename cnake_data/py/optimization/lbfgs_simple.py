"""Simplified L-BFGS optimization.

Keywords: lbfgs, quasi-newton, optimization, minimization, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def lbfgs_simple(n: int) -> float:
    """L-BFGS minimization of sum_i (x_i - i)^2 + 0.1*sum(x_i * x_{i+1}).

    n variables, 100 iterations, m=5 history. Returns final objective value.

    Args:
        n: Number of variables.

    Returns:
        Final objective function value.
    """
    m = 5  # history size

    # Initialize x to zeros
    x = [0.0] * n
    g = [0.0] * n

    # History storage
    s_hist = []  # s_k = x_{k+1} - x_k
    y_hist = []  # y_k = g_{k+1} - g_k
    rho_hist = []

    for _iteration in range(100):
        # Compute gradient
        for i in range(n):
            g[i] = 2.0 * (x[i] - i)
        for i in range(n - 1):
            g[i] += 0.1 * x[i + 1]
            g[i + 1] += 0.1 * x[i]

        # L-BFGS two-loop recursion
        q = [g[i] for i in range(n)]
        hist_len = len(s_hist)
        alpha_list = [0.0] * hist_len

        for j in range(hist_len - 1, -1, -1):
            sj = s_hist[j]
            dot_val = 0.0
            for i in range(n):
                dot_val += sj[i] * q[i]
            alpha_list[j] = rho_hist[j] * dot_val
            yj = y_hist[j]
            for i in range(n):
                q[i] -= alpha_list[j] * yj[i]

        # Scale initial Hessian
        if hist_len > 0:
            sy = 0.0
            yy = 0.0
            sj = s_hist[-1]
            yj = y_hist[-1]
            for i in range(n):
                sy += sj[i] * yj[i]
                yy += yj[i] * yj[i]
            if yy > 0:
                gamma = sy / yy
                for i in range(n):
                    q[i] *= gamma

        for j in range(hist_len):
            yj = y_hist[j]
            dot_val = 0.0
            for i in range(n):
                dot_val += yj[i] * q[i]
            beta = rho_hist[j] * dot_val
            sj = s_hist[j]
            for i in range(n):
                q[i] += (alpha_list[j] - beta) * sj[i]

        # Direction d = -q, line search with fixed step
        step = 0.5
        x_old = [x[i] for i in range(n)]
        g_old = [g[i] for i in range(n)]

        for i in range(n):
            x[i] -= step * q[i]

        # Compute new gradient for history update
        g_new = [0.0] * n
        for i in range(n):
            g_new[i] = 2.0 * (x[i] - i)
        for i in range(n - 1):
            g_new[i] += 0.1 * x[i + 1]
            g_new[i + 1] += 0.1 * x[i]

        # Update history
        sk = [x[i] - x_old[i] for i in range(n)]
        yk = [g_new[i] - g_old[i] for i in range(n)]
        sy_dot = 0.0
        for i in range(n):
            sy_dot += sk[i] * yk[i]

        if abs(sy_dot) > 1e-30:
            if len(s_hist) >= m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(sk)
            y_hist.append(yk)
            rho_hist.append(1.0 / sy_dot)

    # Compute final objective
    obj = 0.0
    for i in range(n):
        diff = x[i] - i
        obj += diff * diff
    for i in range(n - 1):
        obj += 0.1 * x[i] * x[i + 1]

    return obj
