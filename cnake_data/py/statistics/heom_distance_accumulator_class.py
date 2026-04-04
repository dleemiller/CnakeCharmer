"""Class-based mixed-type distance accumulator (HEOM-style).

Keywords: statistics, class, distance metric, mixed features, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class HeomAccumulator:
    def __init__(self, n_cols: int, cat_stride: int):
        self.n_cols = n_cols
        self.cat_stride = cat_stride
        self.is_cat = [1 if (i % cat_stride) == 0 else 0 for i in range(n_cols)]

    def distance_scaled(
        self, row_a: list[int], row_b: list[int], mins: list[int], spans: list[int]
    ) -> int:
        total = 0
        for c in range(self.n_cols):
            if self.is_cat[c]:
                total += 1000 if row_a[c] != row_b[c] else 0
            else:
                da = row_a[c] - row_b[c]
                if da < 0:
                    da = -da
                total += (da * 1000) // spans[c]
        return total


@python_benchmark(args=(550, 18, 4, 23))
def heom_distance_accumulator_class(n_rows: int, n_cols: int, cat_stride: int, seed: int) -> tuple:
    rows = [[0] * n_cols for _ in range(n_rows)]
    mins = [10**9] * n_cols
    maxs = [-(10**9)] * n_cols

    for r in range(n_rows):
        base = seed * 104729 + r * 8191
        for c in range(n_cols):
            v = (base + c * 131 + (r * c) * 17) % 251
            rows[r][c] = v
            if v < mins[c]:
                mins[c] = v
            if v > maxs[c]:
                maxs[c] = v

    spans = [max(1, maxs[c] - mins[c]) for c in range(n_cols)]
    metric = HeomAccumulator(n_cols, cat_stride)

    total = 0
    cat_mismatches = 0
    last = 0
    for r in range(1, n_rows):
        d = metric.distance_scaled(rows[r - 1], rows[r], mins, spans)
        total += d
        last = d
        for c in range(0, n_cols, cat_stride):
            if rows[r - 1][c] != rows[r][c]:
                cat_mismatches += 1

    return (total, cat_mismatches, last)
