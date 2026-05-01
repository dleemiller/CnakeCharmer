"""Simple nearest-label classifier with class prototype compression."""

from __future__ import annotations

import csv
import math


def compress_dataset(
    dataset: list[list[float]], targets: list[list[str]]
) -> tuple[list[list[float]], list[list[str]]]:
    model: dict[str, list[float]] = {}
    counts: dict[str, int] = {}

    for t, d in zip(targets, dataset, strict=False):
        label = t[0]
        if label not in model:
            model[label] = list(d)
            counts[label] = 1
        else:
            for i in range(len(d)):
                model[label][i] += d[i]
            counts[label] += 1

    compressed_data = []
    compressed_targets = []
    for label, data in model.items():
        compressed = [x / float(counts[label]) for x in data]
        compressed_data.append(compressed)
        compressed_targets.append([label])

    return compressed_data, compressed_targets


def load_dataset(path: str) -> tuple[list[list[float]], list[list[str]]]:
    rows = [r for r in csv.reader(open(path, newline=""))]
    dataset = [[float(x) for x in row[0:-1]] for row in rows][:-1]
    targets = [row[-1:] for row in rows][:-1]
    return compress_dataset(dataset, targets)


def magnitude(a: list[float], b: list[float]) -> float:
    distance = 0.0
    for x, y in zip(a, b, strict=False):
        distance += (x - y) ** 2
    return math.sqrt(distance)


def classify(vector: list[float], dataset: list[list[float]], outputs: list[list[str]]):
    if not (vector and dataset and outputs):
        raise ValueError("input data must not be empty")
    if not (len(vector) == len(dataset[0]) and len(dataset) == len(outputs)):
        raise TypeError("data shape must be consistent")

    distances = [magnitude(vector, x) for x in dataset]
    i = distances.index(min(distances))
    return outputs[i]
