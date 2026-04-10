def insertion_sort(n):
    """Sort a deterministic array of n integers using insertion sort.

    Returns (first element, last element, number of swaps).
    """
    arr = [0] * n
    for i in range(n):
        arr[i] = (i * 2654435761) & 0xFFFF

    swaps = 0
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
            swaps += 1
        arr[j + 1] = key

    return (arr[0], arr[-1], swaps)
