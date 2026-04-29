"""Word beam search decoding over CTC-like outputs."""

from __future__ import annotations


def word_beam_search(mat, beam_width, blank_idx, next_chars_fn):
    beams = {"": (1.0, 0.0)}  # text -> (p_blank, p_non_blank)
    for t in range(len(mat)):
        curr = {}
        best = sorted(beams.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)[:beam_width]
        for text, (p_b, p_nb) in best:
            pr_blank = (p_b + p_nb) * mat[t][blank_idx]
            cb, cnb = curr.get(text, (0.0, 0.0))
            curr[text] = (cb + pr_blank, cnb)

            for c in next_chars_fn(text):
                idx = c + 1
                if text and idx - 1 == ord(text[-1]):
                    pr_non_blank = mat[t][idx] * p_b
                else:
                    pr_non_blank = mat[t][idx] * (p_b + p_nb)
                nt = text + chr(c)
                cb2, cnb2 = curr.get(nt, (0.0, 0.0))
                curr[nt] = (cb2, cnb2 + pr_non_blank)
        beams = curr

    return max(beams.items(), key=lambda kv: kv[1][0] + kv[1][1])[0]
