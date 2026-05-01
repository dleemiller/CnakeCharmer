"""CTC beam search decoding core."""

from __future__ import annotations


class BeamEntry:
    def __init__(self, alpha=1.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta
        self.p_blank = 0.0
        self.p_non_blank = 0.0
        self.p_total = 0.0
        self.p_lm = 1.0
        self.is_lm_applied = False
        self.labeling = ()

    @property
    def score(self):
        return (
            self.p_total
            * (self.p_lm**self.alpha)
            * (len(self.labeling) ** self.beta if self.labeling else 1.0)
        )


class BeamState:
    def __init__(self, alpha=1.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta
        self.entries = {}

    @property
    def sorted_entries(self):
        return sorted(self.entries.values(), key=lambda x: x.score, reverse=True)

    def add_beam(self, labeling):
        if labeling not in self.entries:
            self.entries[labeling] = BeamEntry(alpha=self.alpha, beta=self.beta)


def apply_lm(beam, id2char, lm):
    if lm and not beam.is_lm_applied:
        text = "".join(id2char[i] for i in beam.labeling)
        beam.p_lm = lm(text)
        beam.is_lm_applied = True


def ctc_beam_search(mat, classes, lm=None, beam_width=10, alpha=1.0, beta=2.0, min_char_prob=0.001):
    blank_idx = len(classes)
    max_t = len(mat)
    max_c = len(mat[0])

    last = BeamState(alpha=alpha, beta=beta)
    last.entries[()] = BeamEntry(alpha=alpha, beta=beta)
    last.entries[()].p_blank = 1.0
    last.entries[()].p_total = 1.0

    for t in range(max_t):
        curr = BeamState(alpha=alpha, beta=beta)
        for entry in last.sorted_entries[:beam_width]:
            labeling = entry.labeling

            p_non_blank = (
                last.entries[labeling].p_non_blank * mat[t][labeling[-1]] if labeling else 0.0
            )
            p_blank = last.entries[labeling].p_total * mat[t][blank_idx]

            curr.add_beam(labeling)
            be = curr.entries[labeling]
            be.labeling = labeling
            be.p_non_blank += p_non_blank
            be.p_blank += p_blank
            be.p_total += p_blank + p_non_blank
            be.p_lm = last.entries[labeling].p_lm
            be.is_lm_applied = True

            for c in range(max_c - 1):
                if mat[t][c] < min_char_prob:
                    continue
                new_labeling = labeling + (c,)
                if labeling and labeling[-1] == c:
                    p = mat[t][c] * last.entries[labeling].p_blank
                else:
                    p = mat[t][c] * last.entries[labeling].p_total

                curr.add_beam(new_labeling)
                ne = curr.entries[new_labeling]
                ne.labeling = new_labeling
                ne.p_non_blank += p
                ne.p_total += p
                apply_lm(ne, classes, lm)

        last = curr

    return last
