"""N-dimensional Nelder-Mead optimizer.

Keywords: nelder-mead, simplex, optimization, n-dimensional, rosenbrock, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(4, 12))
def nelder_mead(dim: int, n_starts: int) -> tuple:
    """N-dimensional Nelder-Mead minimization of the Rosenbrock function.

    Runs n_starts independent optimizations from deterministic starting points,
    each for 200*dim iterations. Returns (best_value, trajectory_checksum) where
    trajectory_checksum is a hash-like sum of all best values across runs for
    reproducibility checking.

    Args:
        dim: Number of dimensions (>= 2).
        n_starts: Number of independent restarts.

    Returns:
        Tuple of (best_value_found, trajectory_checksum).
    """
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5  # contraction
    sigma = 0.5  # shrink
    max_iter = 200 * dim
    step = 0.5

    global_best = 1e30
    checksum = 0.0

    for s in range(n_starts):
        # Deterministic starting point
        x0 = [0.0] * dim
        for d in range(dim):
            x0[d] = ((s * 137 + d * 43) % 1000) / 250.0 - 2.0

        # Build initial simplex: dim+1 vertices
        n_verts = dim + 1
        simplex = [[0.0] * dim for _ in range(n_verts)]
        f_vals = [0.0] * n_verts

        # First vertex is starting point
        for d in range(dim):
            simplex[0][d] = x0[d]
        f_vals[0] = _rosenbrock(simplex[0], dim)

        # Remaining vertices: perturb one coordinate
        for i in range(1, n_verts):
            for d in range(dim):
                simplex[i][d] = x0[d]
            simplex[i][i - 1] = x0[i - 1] + step
            f_vals[i] = _rosenbrock(simplex[i], dim)

        for _it in range(max_iter):
            # Sort simplex by function value (insertion sort)
            for i in range(1, n_verts):
                key_f = f_vals[i]
                key_v = simplex[i]
                j = i - 1
                while j >= 0 and f_vals[j] > key_f:
                    f_vals[j + 1] = f_vals[j]
                    simplex[j + 1] = simplex[j]
                    j -= 1
                f_vals[j + 1] = key_f
                simplex[j + 1] = key_v

            # Centroid of all but worst
            centroid = [0.0] * dim
            for i in range(n_verts - 1):
                for d in range(dim):
                    centroid[d] += simplex[i][d]
            inv_n = 1.0 / dim  # n_verts-1 == dim
            for d in range(dim):
                centroid[d] *= inv_n

            worst = simplex[dim]
            f_worst = f_vals[dim]

            # Reflection
            xr = [0.0] * dim
            for d in range(dim):
                xr[d] = centroid[d] + alpha * (centroid[d] - worst[d])
            f_r = _rosenbrock(xr, dim)

            if f_vals[0] <= f_r < f_vals[dim - 1]:
                simplex[dim] = xr
                f_vals[dim] = f_r
                continue

            # Expansion
            if f_r < f_vals[0]:
                xe = [0.0] * dim
                for d in range(dim):
                    xe[d] = centroid[d] + gamma * (xr[d] - centroid[d])
                f_e = _rosenbrock(xe, dim)
                if f_e < f_r:
                    simplex[dim] = xe
                    f_vals[dim] = f_e
                else:
                    simplex[dim] = xr
                    f_vals[dim] = f_r
                continue

            # Contraction
            if f_r < f_worst:
                # Outside contraction
                xc = [0.0] * dim
                for d in range(dim):
                    xc[d] = centroid[d] + rho * (xr[d] - centroid[d])
                f_c = _rosenbrock(xc, dim)
                if f_c <= f_r:
                    simplex[dim] = xc
                    f_vals[dim] = f_c
                    continue
            else:
                # Inside contraction
                xc = [0.0] * dim
                for d in range(dim):
                    xc[d] = centroid[d] + rho * (worst[d] - centroid[d])
                f_c = _rosenbrock(xc, dim)
                if f_c < f_worst:
                    simplex[dim] = xc
                    f_vals[dim] = f_c
                    continue

            # Shrink
            best_v = simplex[0]
            for i in range(1, n_verts):
                for d in range(dim):
                    simplex[i][d] = best_v[d] + sigma * (simplex[i][d] - best_v[d])
                f_vals[i] = _rosenbrock(simplex[i], dim)

        best_f = f_vals[0]
        checksum += best_f * (s + 1)
        if best_f < global_best:
            global_best = best_f

    return (global_best, checksum)


def _rosenbrock(x, dim):
    """Rosenbrock function in N dimensions."""
    val = 0.0
    for i in range(dim - 1):
        t1 = x[i + 1] - x[i] * x[i]
        t2 = 1.0 - x[i]
        val += 100.0 * t1 * t1 + t2 * t2
    return val
