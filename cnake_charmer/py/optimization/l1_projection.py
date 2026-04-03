"""L1 ball projection via sort-and-threshold.

Projects each deterministic vector onto the L1 ball of a given radius
by sorting absolute values, finding the soft-threshold, and shrinking.

Keywords: optimization, projection, l1 norm, soft threshold, sorting, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(80000,))
def l1_projection(n: int) -> tuple:
    """Project n deterministic vectors of length 12 onto the L1 ball.

    For each vector, we sort the absolute values in descending order,
    find the threshold theta such that sum(max(|x_i| - theta, 0)) = radius,
    then soft-threshold each component.

    Args:
        n: Number of vectors to process.

    Returns:
        Tuple of (result_sum, max_abs_after) over all projected vectors.
    """
    radius = 3.0
    dim = 12
    result_sum = 0.0
    max_abs_after = 0.0

    for k in range(n):
        # Generate deterministic vector with varied magnitudes
        vec = [0.0] * dim
        for d in range(dim):
            vec[d] = ((k * 17 + d * 31 + 5) % 2003) / 200.0 - 5.0

        # Compute L1 norm
        l1_norm = 0.0
        for d in range(dim):
            v = vec[d]
            if v < 0.0:
                l1_norm -= v
            else:
                l1_norm += v

        # If already inside the L1 ball, no projection needed
        if l1_norm <= radius:
            for d in range(dim):
                result_sum += vec[d]
                av = vec[d]
                if av < 0.0:
                    av = -av
                if av > max_abs_after:
                    max_abs_after = av
            continue

        # Sort absolute values descending (manual insertion sort)
        abs_vals = [0.0] * dim
        for d in range(dim):
            v = vec[d]
            if v < 0.0:
                abs_vals[d] = -v
            else:
                abs_vals[d] = v

        # Insertion sort descending
        for i in range(1, dim):
            key = abs_vals[i]
            j = i - 1
            while j >= 0 and abs_vals[j] < key:
                abs_vals[j + 1] = abs_vals[j]
                j -= 1
            abs_vals[j + 1] = key

        # Find theta: the soft-threshold value
        # We need sum_{i: |x_i| > theta} (|x_i| - theta) = radius
        # Walk sorted values to find the breakpoint
        cumsum = 0.0
        theta = 0.0
        for i in range(dim):
            cumsum += abs_vals[i]
            candidate = (cumsum - radius) / (i + 1)
            if i < dim - 1 and candidate >= abs_vals[i + 1]:
                theta = candidate
                break
            if i == dim - 1:
                theta = candidate

        # Apply soft-thresholding: sign(x_i) * max(|x_i| - theta, 0)
        for d in range(dim):
            val = vec[d]
            av = -val if val < 0.0 else val

            if av <= theta:
                shrunk = 0.0
            elif val > 0.0:
                shrunk = val - theta
            else:
                shrunk = val + theta

            result_sum += shrunk
            av_after = -shrunk if shrunk < 0.0 else shrunk
            if av_after > max_abs_after:
                max_abs_after = av_after

    return (result_sum, max_abs_after)
