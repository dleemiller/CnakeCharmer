"""Class-based sorted storage with nearest-value queries.

Keywords: algorithms, class, sorted storage, binary search, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class SortedCellStorage:
    def __init__(self):
        self.values: list[int] = []

    def insert(self, value: int) -> None:
        lo = 0
        hi = len(self.values)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.values[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        self.values.insert(lo, value)

    def nearest(self, target: int) -> int:
        n = len(self.values)
        if n == 0:
            return 0
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.values[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return self.values[0]
        if lo == n:
            return self.values[-1]
        a = self.values[lo - 1]
        b = self.values[lo]
        da = target - a
        db = b - target
        return a if da <= db else b


@python_benchmark(args=(5000, 220000, 29, 100003))
def sorted_cell_storage_class(n_values: int, n_queries: int, seed: int, mod: int) -> tuple:
    st = SortedCellStorage()
    for i in range(n_values):
        v = (seed * 2654435761 + i * 40507 + (i % 11) * 97) % mod
        st.insert(v)

    s = 0
    near_sum = 0
    for q in range(n_queries):
        t = (seed * 1664525 + q * 1013904223 + s) % mod
        n = st.nearest(t)
        near_sum += n
        s = (s + (n ^ t)) & 0xFFFFFFFF

    return (len(st.values), near_sum & 0xFFFFFFFF, s)
