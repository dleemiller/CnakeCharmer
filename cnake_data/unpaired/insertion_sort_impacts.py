"""In-place insertion sort for impact scores."""

from __future__ import annotations


def sort_impacts_based_on_comparison(impacts):
    for i in range(1, len(impacts)):
        k = impacts[i]
        j = i - 1
        while j >= 0 and k < impacts[j]:
            impacts[j + 1] = impacts[j]
            j -= 1
        impacts[j + 1] = k
    return impacts
