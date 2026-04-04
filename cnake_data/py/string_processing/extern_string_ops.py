"""Compute total string length and equal pair count for generated strings.

Keywords: string processing, extern, strlen, strcmp, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def extern_string_ops(n: int) -> int:
    """Generate n short strings, compute total length and equal pairs.

    Args:
        n: Number of strings.

    Returns:
        Sum of all string lengths plus count of adjacent equal pairs.
    """
    strings = []
    for i in range(n):
        seed = i * 2654435761 + 17
        length = (seed & 0x7FFFFFFF) % 8 + 1
        chars = []
        for _j in range(length):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            chars.append(chr(65 + seed % 26))
        strings.append("".join(chars))

    length_sum = 0
    for s in strings:
        length_sum += len(s)

    equal_count = 0
    for i in range(1, n):
        if strings[i] == strings[i - 1]:
            equal_count += 1

    return length_sum + equal_count
