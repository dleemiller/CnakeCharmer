"""Quality trimming routines for sequencing reads."""

from __future__ import annotations


def quality_trim_index(qualities: str, cutoff_front: int, cutoff_back: int, base: int = 33):
    stop = len(qualities)
    start = 0

    s = 0
    max_qual = 0
    for i in range(len(qualities)):
        s += cutoff_front - (ord(qualities[i]) - base)
        if s < 0:
            break
        if s > max_qual:
            max_qual = s
            start = i + 1

    max_qual = 0
    s = 0
    for i in range(len(qualities) - 1, -1, -1):
        s += cutoff_back - (ord(qualities[i]) - base)
        if s < 0:
            break
        if s > max_qual:
            max_qual = s
            stop = i

    if start >= stop:
        start, stop = 0, 0
    return start, stop


def nextseq_trim_index(sequence, cutoff: int, base: int = 33):
    bases = sequence.sequence
    qualities = sequence.qualities
    s = 0
    max_qual = 0
    max_i = len(qualities)
    for i in range(max_i - 1, -1, -1):
        q = ord(qualities[i]) - base
        if bases[i] == "G":
            q = cutoff - 1
        s += cutoff - q
        if s < 0:
            break
        if s > max_qual:
            max_qual = s
            max_i = i
    return max_i
