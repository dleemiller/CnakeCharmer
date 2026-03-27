# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# distutils: language = c++
"""Max-heap with periodic draining using C++ std::priority_queue.

Keywords: algorithms, heap, priority queue, max-heap, std::priority_queue, drain, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "<queue>" namespace "std" nogil:
    cdef cppclass priority_queue[T]:
        priority_queue()
        void push(T)
        T& top()
        void pop()
        bint empty()
        size_t size()


@cython_benchmark(syntax="cy", args=(500000,))
def cpp_priority_queue_drain(int n):
    """Push n values into a C++ max-heap; drain top 10 every 100 pushes.

    Values pushed: (i * 2654435761) % 1000000
    Every 100 pushes, drain up to 10 maximum elements and accumulate their sum.

    Args:
        n: Total number of values to push.

    Returns:
        Tuple of (total_drained_sum, drain_count) where drain_count is the
        number of drain events performed.
    """
    cdef priority_queue[int] pq
    cdef int i, j, val
    cdef long long total_drained_sum = 0
    cdef int drain_count = 0

    for i in range(n):
        val = <int>((<long long>i * <long long>2654435761) % 1000000)
        pq.push(val)
        if (i + 1) % 100 == 0:
            for j in range(10):
                if not pq.empty():
                    total_drained_sum += pq.top()
                    pq.pop()
            drain_count += 1

    return (total_drained_sum, drain_count)
