import math
from operator import itemgetter


class CharacterTrigramFuzzySet:
    def __init__(self, iterable, max_count=100, relative_threshold=0.5):
        self._match_dict = {}
        self._items = []
        self._seen = set()
        self._max_count = int(max_count)
        self._relative_threshold = float(relative_threshold)
        for value in iterable:
            self._add(value)

    def _add(self, value):
        lvalue = value.lower()
        if lvalue in self._seen:
            return
        self._seen.add(lvalue)

        simplified = "-" + lvalue + "-"
        gram_count = len(simplified) - 2
        idx = len(self._items)

        for i in range(gram_count):
            gram = simplified[i : i + 3]
            self._match_dict.setdefault(gram, []).append(idx)

        self._items.append((math.sqrt(gram_count), lvalue))

    def get(self, value):
        simplified = "-" + value.lower() + "-"
        gram_count = len(simplified) - 2
        norm = math.sqrt(gram_count)

        matches = {}
        for i in range(gram_count):
            for idx in self._match_dict.get(simplified[i : i + 3], ()):
                matches[idx] = matches.get(idx, 0) + 1

        results = [
            (score / (norm * self._items[idx][0]), self._items[idx][1])
            for idx, score in matches.items()
        ]

        if not results:
            return []

        results.sort(reverse=True, key=itemgetter(0))
        threshold = results[0][0] * self._relative_threshold
        return [(score, word) for score, word in results[: self._max_count] if score > threshold]
