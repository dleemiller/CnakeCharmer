from __future__ import annotations


def heapify(values: list[float]) -> list[float]:
    h = values[:]
    n = len(h)
    for i in range(n // 2 - 1, -1, -1):
        j = i
        while True:
            left = 2 * j + 1
            right = left + 1
            smallest = j
            if left < n and h[left] < h[smallest]:
                smallest = left
            if right < n and h[right] < h[smallest]:
                smallest = right
            if smallest == j:
                break
            h[j], h[smallest] = h[smallest], h[j]
            j = smallest
    return h


def decrease_key(heap: list[float], idx: int, new_val: float) -> None:
    if new_val > heap[idx]:
        raise ValueError("new_val must be <= current value")
    heap[idx] = new_val
    i = idx
    while i > 0:
        p = (i - 1) // 2
        if heap[p] <= heap[i]:
            break
        heap[p], heap[i] = heap[i], heap[p]
        i = p
