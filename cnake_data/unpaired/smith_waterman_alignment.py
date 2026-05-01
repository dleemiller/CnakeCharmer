"""Smith-Waterman local alignment with traceback."""

from __future__ import annotations


def smith_waterman(
    read: str, sequence: str, match_score: int = 1, gap_score: int = -2
) -> list[tuple[str, str, int]]:
    read_len = len(read)
    seq_len = len(sequence)
    width = seq_len + 1
    dim = (read_len + 1) * width
    dp = [0] * dim

    max_score = 0
    max_coords: list[int] = []

    for i in range(1, read_len + 1):
        row = i * width
        for j in range(1, seq_len + 1):
            cur = row + j
            left = cur - 1
            above = cur - width
            diag = above - 1

            diag_score = dp[diag] + (
                match_score if read[i - 1] == sequence[j - 1] else -match_score
            )
            deletion = dp[left] + gap_score
            insertion = dp[above] + gap_score
            best = max(diag_score, deletion, insertion, 0)
            dp[cur] = best

            if best > max_score:
                max_score = best
                max_coords = [cur]
            elif best == max_score and best > 0:
                max_coords.append(cur)

    alignments: list[tuple[str, str, int]] = []
    for coord0 in max_coords:
        coord = coord0
        score = dp[coord]
        out_read = []
        out_seq = []
        while dp[coord] != 0:
            i = coord // width
            j = coord % width
            left = coord - 1
            above = coord - width
            diag = above - 1
            if score == dp[diag] + (
                match_score if read[i - 1] == sequence[j - 1] else -match_score
            ):
                out_read.append(read[i - 1])
                out_seq.append(sequence[j - 1])
                coord = diag
            elif score == dp[left] + gap_score:
                out_read.append("-")
                out_seq.append(sequence[j - 1])
                coord = left
            else:
                out_read.append(read[i - 1])
                out_seq.append("-")
                coord = above
            score = dp[coord]

        alignments.append(("".join(reversed(out_read)), "".join(reversed(out_seq)), max_score))

    return alignments
