"""L-infinity proximal operator applied to deterministic vectors.

Computes the proximal operator of the L-infinity norm: projects each
vector onto the L1 ball of radius tau by soft-thresholding sorted
absolute values.

Keywords: optimization, proximal, l-infinity, projection, sorting, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def linf_proximal(n: int) -> tuple:
    """Apply L-inf proximal operator to n deterministic vectors of length 8.

    For each vector, sort absolute values descending, find the threshold
    that projects onto the L1 ball, then soft-threshold the original values.

    Args:
        n: Number of vectors to process.

    Returns:
        Tuple of (result_sum, max_abs_after) where result_sum is the sum of
        all output elements and max_abs_after is the largest absolute value
        across all outputs.
    """
    tau = 1.5
    dim = 8
    result_sum = 0.0
    max_abs_after = 0.0

    for k in range(n):
        # Generate deterministic vector
        vec = [0.0] * dim
        for d in range(dim):
            vec[d] = ((k * 7 + d * 13 + 3) % 1000) / 100.0 - 5.0

        # Sort absolute values descending
        abs_vals = [abs(v) for v in vec]
        abs_vals.sort(reverse=True)

        # Find threshold: largest lambda such that
        # sum(max(|x_i| - lambda, 0)) <= tau
        lam = 0.0
        cumsum = 0.0
        for i in range(dim):
            cumsum += abs_vals[i]
            candidate = (cumsum - tau) / (i + 1)
            if candidate > abs_vals[i]:
                break
            if candidate > lam:
                lam = candidate

        # Apply soft-thresholding
        for d in range(dim):
            val = vec[d]
            av = abs(val)
            if av <= lam:
                shrunk = 0.0
            elif val > 0:
                shrunk = val - lam
            else:
                shrunk = val + lam

            result_sum += shrunk
            av_after = abs(shrunk)
            if av_after > max_abs_after:
                max_abs_after = av_after

    return (result_sum, max_abs_after)
