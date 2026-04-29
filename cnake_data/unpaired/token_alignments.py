"""Token alignment between two differently tokenized strings."""

from __future__ import annotations

import re
from itertools import chain


def get_alignments(a_tokens, b_tokens):
    char_to_a = tuple(chain(*((i,) * len(x.lower()) for i, x in enumerate(a_tokens))))
    char_to_b = tuple(chain(*((i,) * len(x.lower()) for i, x in enumerate(b_tokens))))
    sa = "".join(a_tokens).lower()
    sb = "".join(b_tokens).lower()

    if re.sub(r"\s+", "", sa) != re.sub(r"\s+", "", sb):
        raise ValueError("texts differ beyond whitespace/case")

    a2b, b2a = [], []
    ca = cb = 0
    prev_a = prev_b = -1

    while ca < len(sa) and cb < len(sb):
        ta = char_to_a[ca]
        tb = char_to_b[cb]
        if prev_a != ta:
            a2b.append(set())
        if prev_b != tb:
            b2a.append(set())

        if sa[ca] == sb[cb]:
            a2b[-1].add(tb)
            b2a[-1].add(ta)
            ca += 1
            cb += 1
        elif sa[ca].isspace():
            ca += 1
        elif sb[cb].isspace():
            cb += 1
        else:
            raise ValueError("unexpected mismatch")

        prev_a, prev_b = ta, tb

    a2b.extend([set()] * len(set(char_to_a[ca:])))
    b2a.extend([set()] * len(set(char_to_b[cb:])))
    return [sorted(x) for x in a2b], [sorted(x) for x in b2a]
