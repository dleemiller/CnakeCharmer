"""Class-based barcode index with Hamming-distance binning.

Keywords: string_processing, class, barcode, hamming, matching, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class BarcodeIndex:
    def __init__(self, n_codes: int, code_len: int, seed: int):
        self.codes = [[0] * code_len for _ in range(n_codes)]
        for i in range(n_codes):
            x = (seed + i * 37) & 0x7FFFFFFF
            for j in range(code_len):
                x = (1103515245 * x + 12345 + j * 17) & 0x7FFFFFFF
                self.codes[i][j] = x & 3

    def mutate_code(self, src: list[int], edits: int, salt: int) -> list[int]:
        out = src[:]
        n = len(out)
        for k in range(edits):
            idx = (salt * 131 + k * 17) % n
            out[idx] = (out[idx] + 1 + ((salt + k) & 1)) & 3
        return out

    def hamming(self, a: list[int], b: list[int]) -> int:
        d = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                d += 1
        return d


@python_benchmark(args=(420, 28, 3, 7, 41))
def barcode_hamming_bins_class(
    n_codes: int,
    code_len: int,
    edits: int,
    threshold: int,
    seed: int,
) -> tuple:
    idx = BarcodeIndex(n_codes, code_len, seed)
    matches = 0
    total_dist = 0
    nearest_sum = 0

    for i in range(n_codes):
        probe = idx.mutate_code(idx.codes[i], edits, seed + i * 19)
        best = code_len + 1
        best_j = -1
        for j in range(n_codes):
            d = idx.hamming(probe, idx.codes[j])
            total_dist += d
            if d < best:
                best = d
                best_j = j
        if best <= threshold:
            matches += 1
        nearest_sum += best_j

    return (matches, total_dist, nearest_sum)
