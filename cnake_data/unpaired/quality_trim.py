"""Quality trimming index finder (BWA-style partial-sum heuristic)."""

from __future__ import annotations


def quality_trim_index(qualities, cutoff_front, cutoff_back, base=33):
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
