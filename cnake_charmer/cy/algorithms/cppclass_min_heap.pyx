# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# distutils: language = c++
"""Min-heap priority queue using an inline C++ class wrapping std::priority_queue.

Keywords: heap, priority queue, min-heap, cppclass, cdef extern, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

cdef extern from *:
    """
    #include <queue>
    #include <vector>
    #include <functional>
    struct MinHeap {
        std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
        void push(int val) { pq.push(val); }
        int top() { return pq.top(); }
        void pop() { pq.pop(); }
        bool empty() { return pq.empty(); }
        int size() { return (int)pq.size(); }
    };
    """
    cdef cppclass MinHeap:
        void push(int val)
        int top()
        void pop()
        bint empty()
        int size()


@cython_benchmark(syntax="cy", args=(500000,))
def cppclass_min_heap(int n):
    """Process n heap operations: push phase then alternating pop/push.

    First 3n/4 operations are pushes of (i * 2654435761) % 1_000_000.
    Remaining n/4 operations alternate pop_min and push of a new value.

    Args:
        n: Total number of operations.

    Returns:
        Tuple of (sum_of_popped_values, pop_count).
    """
    cdef MinHeap heap
    cdef int i, j, push_count, remaining, val
    cdef long long sum_popped = 0
    cdef int pop_count = 0

    push_count = (n * 3) // 4
    remaining = n - push_count

    for i in range(push_count):
        val = <int>((<long long>i * <long long>2654435761) % <long long>1000000)
        heap.push(val)

    for j in range(remaining):
        if j % 2 == 0:
            if not heap.empty():
                sum_popped += heap.top()
                heap.pop()
                pop_count += 1
        else:
            val = <int>((<long long>(push_count + j) * <long long>2654435761) % <long long>1000000)
            heap.push(val)

    return (<long long>sum_popped, pop_count)
