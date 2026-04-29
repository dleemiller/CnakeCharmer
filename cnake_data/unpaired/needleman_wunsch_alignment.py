"""Needleman-Wunsch global alignment."""

from __future__ import annotations

from dataclasses import dataclass

NO_POINTER = 2
POINTER_UP = -1
POINTER_LEFT = 1
POINTER_DIAG = 0
GAP = "-"


@dataclass
class NeedlemanWunsch:
    match: float = 1.0
    mismatch: float = -1.0
    gap: float = -1.0

    def align(self, seq1: str, seq2: str) -> tuple[str, str]:
        n, m = len(seq2), len(seq1)
        p = [[NO_POINTER] * (m + 1) for _ in range(n + 1)]
        a = [[0.0] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            p[i][0] = POINTER_UP
            a[i][0] = a[i - 1][0] + self.gap
        for j in range(1, m + 1):
            p[0][j] = POINTER_LEFT
            a[0][j] = a[0][j - 1] + self.gap

        for i in range(1, n + 1):
            c2 = seq2[i - 1]
            for j in range(1, m + 1):
                top = a[i - 1][j] + self.gap
                left = a[i][j - 1] + self.gap
                diag = a[i - 1][j - 1] + (self.match if seq1[j - 1] == c2 else self.mismatch)
                if diag >= top and diag >= left:
                    p[i][j] = POINTER_DIAG
                    a[i][j] = diag
                elif top > left:
                    p[i][j] = POINTER_UP
                    a[i][j] = top
                else:
                    p[i][j] = POINTER_LEFT
                    a[i][j] = left

        i, j = n, m
        out1: list[str] = []
        out2: list[str] = []
        while True:
            pointer = p[i][j]
            if pointer == POINTER_DIAG:
                out1.append(seq1[j - 1])
                out2.append(seq2[i - 1])
                i -= 1
                j -= 1
            elif pointer == POINTER_UP:
                out1.append(GAP)
                out2.append(seq2[i - 1])
                i -= 1
            elif pointer == POINTER_LEFT:
                out1.append(seq1[j - 1])
                out2.append(GAP)
                j -= 1
            else:
                break

        out1.reverse()
        out2.reverse()
        return "".join(out1), "".join(out2)
