"""Parse comma-separated mixed tokens and summarize numeric invoices.

Adapted from The Stack v2 Cython candidate:
- blob_id: d24c7ac24a0676b6e56e0afebdde3fd57041433e
- filename: proWrd.pyx

Keywords: string_processing, parsing, invoice, digit tokens, csv
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def stack_invoice_digit_sum(n_fields: int) -> tuple:
    """Generate mixed invoice fields and aggregate numeric-only entries."""
    parts = []
    state = 42424242
    for _ in range(n_fields):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        if (state & 3) < 2:
            parts.append(str((state >> 8) % 100000))
        else:
            parts.append("A" + str((state >> 11) % 1000) + "B")

    text = ",".join(parts)
    count = 0
    total = 0
    first = -1
    last = -1

    for tok in text.split(","):
        if tok.isdigit():
            v = int(tok)
            if count == 0:
                first = v
            last = v
            count += 1
            total += v

    return (count, total & 0xFFFFFFFF, first, last)
