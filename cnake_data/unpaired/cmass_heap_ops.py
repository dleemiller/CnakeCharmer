from __future__ import annotations


def heap_push(heap: list[tuple[float, int]], item: tuple[float, int]) -> None:
    heap.append(item)
    i = len(heap) - 1
    while i > 0:
        p = (i - 1) // 2
        if heap[p][0] <= heap[i][0]:
            break
        heap[p], heap[i] = heap[i], heap[p]
        i = p


def heap_pop(heap: list[tuple[float, int]]) -> tuple[float, int]:
    if not heap:
        raise IndexError("pop from empty heap")
    out = heap[0]
    last = heap.pop()
    if heap:
        heap[0] = last
        i = 0
        n = len(heap)
        while True:
            l = 2 * i + 1
            r = l + 1
            if l >= n:
                break
            m = l
            if r < n and heap[r][0] < heap[l][0]:
                m = r
            if heap[i][0] <= heap[m][0]:
                break
            heap[i], heap[m] = heap[m], heap[i]
            i = m
    return out
