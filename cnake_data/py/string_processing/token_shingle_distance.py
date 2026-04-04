"""Token shingle binary cosine and jaccard distances.

Keywords: string_processing, token, shingles, cosine, jaccard
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(240000, 3, 26, 17))
def token_shingle_distance(length: int, ngram: int, vocab: int, seed: int) -> tuple:
    state = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    a = [0] * length
    b = [0] * length
    for i in range(length):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        t = int(state % vocab)
        a[i] = t
        # controlled perturbation
        if (state & 31) == 0:
            b[i] = (t + 1) % vocab
        else:
            b[i] = t

    size = 32768
    mask = size - 1
    ca = [0] * size
    cb = [0] * size

    h1 = 2166136261
    h2 = 2166136261
    mul = 16777619

    for i in range(ngram):
        h1 = ((h1 ^ (a[i] + 1)) * mul) & 0xFFFFFFFF
        h2 = ((h2 ^ (b[i] + 1)) * mul) & 0xFFFFFFFF
    ca[h1 & mask] += 1
    cb[h2 & mask] += 1

    for i in range(ngram, length):
        h1 = ((h1 ^ (a[i] + 1) ^ ((a[i - ngram] + 1) << 1)) * mul) & 0xFFFFFFFF
        h2 = ((h2 ^ (b[i] + 1) ^ ((b[i - ngram] + 1) << 1)) * mul) & 0xFFFFFFFF
        ca[h1 & mask] += 1
        cb[h2 & mask] += 1

    inter = 0
    union = 0
    na = 0
    nb = 0
    dot = 0
    sa = 0
    sb = 0
    for i in range(size):
        xa = ca[i]
        xb = cb[i]
        if xa > 0:
            na += 1
        if xb > 0:
            nb += 1
        if xa > 0 and xb > 0:
            inter += 1
        if xa > 0 or xb > 0:
            union += 1
        dot += xa * xb
        sa += xa * xa
        sb += xb * xb

    binary_cos = inter / ((na * nb) ** 0.5) if na and nb else 0.0
    jaccard = inter / union if union else 0.0
    weighted_cos = dot / ((sa * sb) ** 0.5) if sa and sb else 0.0
    return (binary_cos, jaccard, weighted_cos)
