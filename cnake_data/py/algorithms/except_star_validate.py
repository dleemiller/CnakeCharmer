"""Validate arrays and count valid ones.

Demonstrates error return spec pattern: validation function
raises ValueError on invalid data, caller counts valid arrays.

Keywords: algorithms, validation, error handling, except star, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def except_star_validate(n: int) -> int:
    """Validate n arrays, count how many pass validation.

    An array is invalid if any element is negative or if the
    sum exceeds a threshold.

    Args:
        n: Number of arrays to validate.

    Returns:
        Count of valid arrays.
    """
    mask = 0xFFFFFFFF
    arr_size = 8
    threshold = 3000
    valid_count = 0

    for i in range(n):
        arr = [0] * arr_size
        for k in range(arr_size):
            idx = i * arr_size + k
            h = ((idx * 2654435761) & mask) ^ ((idx * 2246822519) & mask)
            arr[k] = (h & 0xFFF) - 500  # Range [-500, 3595]

        try:
            # Validate: check no negatives and sum < threshold
            total = 0
            for k in range(arr_size):
                if arr[k] < 0:
                    raise ValueError("negative element")
                total += arr[k]
            if total > threshold:
                raise ValueError("sum exceeds threshold")
            valid_count += 1
        except ValueError:
            pass

    return valid_count
