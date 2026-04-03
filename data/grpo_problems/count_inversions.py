def count_inversions(n):
    """Count the number of inversions in a deterministic array using brute force.

    An inversion is a pair (i, j) where i < j but arr[i] > arr[j].
    Returns (inversion_count, max_element, checksum).
    """
    arr = [0] * n
    for i in range(n):
        arr[i] = (i * 2654435761) & 0xFFFF

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1

    max_elem = 0
    checksum = 0
    for v in arr:
        if v > max_elem:
            max_elem = v
        checksum = (checksum + v) & 0xFFFFFFFF

    return (count, max_elem, checksum)
