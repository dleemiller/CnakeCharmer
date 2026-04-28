"""Token-sequence LZ20-like compressor."""

from __future__ import annotations


def glue_seq_s(seq):
    s = seq.split()
    h = " ".join(s[:-1])
    t = s[-1]
    return h, t


def glue_seq_l(seq):
    return " ".join(seq)


def lz20(data):
    data = list(map(str, data))
    data = [ch.replace(" ", "__") for ch in data]

    i = 1
    start = 0
    n_data = len(data)
    seq_dict = {}
    sequences = []

    while i <= n_data:
        dst = data[start:i]
        s = glue_seq_l(dst)
        ls = len(s.split())
        if s in seq_dict:
            if ls + start >= n_data:
                sequences.append(s)
                seq_dict[s] += 1
                break
            seq_dict[s] += 1
            i += 1
            continue

        seq_dict[s] = 1
        if " " in s:
            h, t = glue_seq_s(s)
            sequences.append(h)
            sequences.append(t)
        else:
            sequences.append(s)
        start += ls
        i += 1

    return sequences
