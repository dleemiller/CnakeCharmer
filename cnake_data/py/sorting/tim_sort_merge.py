"""Simulate the merge phase of Tim sort on a generated array.

Keywords: tim sort, merge, sorting, runs, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def tim_sort_merge(n: int) -> list:
    """Sort an array using Tim sort's merge strategy.

    Generates arr[i] = (i*31 + 17) % n, splits into runs of size 32,
    sorts each run with insertion sort, then merges runs pairwise.

    Args:
        n: Size of array to sort.

    Returns:
        Sorted list of integers.
    """
    # Generate array
    arr = [(i * 31 + 17) % n for i in range(n)]

    run_size = 32

    # Insertion sort each run
    for start in range(0, n, run_size):
        end = min(start + run_size, n)
        for i in range(start + 1, end):
            key = arr[i]
            j = i - 1
            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    # Merge runs, doubling size each pass
    size = run_size
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(left + size, n)
            right = min(left + 2 * size, n)
            if mid < right:
                # Merge arr[left:mid] and arr[mid:right]
                merged = []
                ii = left
                jj = mid
                while ii < mid and jj < right:
                    if arr[ii] <= arr[jj]:
                        merged.append(arr[ii])
                        ii += 1
                    else:
                        merged.append(arr[jj])
                        jj += 1
                while ii < mid:
                    merged.append(arr[ii])
                    ii += 1
                while jj < right:
                    merged.append(arr[jj])
                    jj += 1
                for k in range(len(merged)):
                    arr[left + k] = merged[k]
        size *= 2

    return arr
