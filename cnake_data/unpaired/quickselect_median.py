"""In-place quickselect median for 1D float arrays."""

from __future__ import annotations


def _swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def quickselect(arr, k):
    l = 0
    ir = len(arr) - 1
    while True:
        if ir <= l + 1:
            if ir == l + 1 and arr[ir] < arr[l]:
                _swap(arr, l, ir)
            return arr[k]

        mid = (l + ir) >> 1
        _swap(arr, mid, l + 1)
        if arr[l] > arr[ir]:
            _swap(arr, l, ir)
        if arr[l + 1] > arr[ir]:
            _swap(arr, l + 1, ir)
        if arr[l] > arr[l + 1]:
            _swap(arr, l, l + 1)

        i = l + 1
        j = ir
        a = arr[l + 1]
        while True:
            i += 1
            while arr[i] < a:
                i += 1
            j -= 1
            while arr[j] > a:
                j -= 1
            if j < i:
                break
            _swap(arr, i, j)

        arr[l + 1] = arr[j]
        arr[j] = a

        if j >= k:
            ir = j - 1
        if j <= k:
            l = i


def median(array):
    arr = list(array)
    n = len(arr)
    mid = n // 2
    quickselect(arr, mid)
    if n % 2 != 0:
        return arr[mid]
    max_val = arr[0]
    for i in range(1, mid):
        if arr[i] > max_val:
            max_val = arr[i]
    return (arr[mid] + max_val) / 2.0
