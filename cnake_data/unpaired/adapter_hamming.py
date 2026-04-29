"""Adapter matching with Hamming-distance scans."""

from __future__ import annotations


def hamming_distance(read, adapter, start=0):
    compare_length = min(len(adapter), len(read) - start)
    mismatches = 0
    for i in range(compare_length):
        if read[start + i] != adapter[i]:
            mismatches += 1
    return mismatches


def hamming_with_n(ref, seq):
    mismatches = 0
    compare_length = min(len(ref), len(seq))
    for i in range(compare_length):
        if ref[i] != "N" and ref[i] != seq[i]:
            mismatches += 1
    return mismatches


def find_adapter_positions(read, adapter, min_comparison_length, max_distance):
    max_start = len(read) - min_comparison_length
    positions = []
    for start in range(max_start + 1):
        dist = hamming_distance(read, adapter, start)
        if dist <= max_distance:
            positions.append(start)
    return positions
