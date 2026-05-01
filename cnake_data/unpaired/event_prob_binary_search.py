from __future__ import annotations


def bsearch_prob(prob: float, cumulative_event_probs: list[float]) -> int:
    nevents = len(cumulative_event_probs)
    if nevents < 1:
        return -1
    final_event = nevents - 1
    if cumulative_event_probs[0] > prob:
        return 0
    if cumulative_event_probs[final_event] < prob:
        return final_event

    lo = 0
    hi = final_event
    while (hi - lo) > 1:
        mid = (hi + lo) // 2
        half_prob = cumulative_event_probs[mid]
        if half_prob > prob:
            hi = mid
        elif half_prob < prob:
            lo = mid
        else:
            return mid
    return hi
