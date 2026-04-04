"""Class-based card/hand transition metrics with deterministic draws.

Keywords: algorithms, class, cards, hand metrics, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Card:
    def __init__(self, value: int, suit: int):
        self.value = value
        self.suit = suit


class Hand:
    def __init__(self, a: Card, b: Card):
        self.a = a
        self.b = b

    def is_pair(self) -> int:
        return 1 if self.a.value == self.b.value else 0

    def is_suited(self) -> int:
        return 1 if self.a.suit == self.b.suit else 0

    def gap(self) -> int:
        g = self.a.value - self.b.value
        return -g if g < 0 else g

    def score(self) -> int:
        return self.a.value * 17 + self.b.value * 13 + self.a.suit * 7 + self.b.suit * 5


@python_benchmark(args=(140000, 91, 13, 4))
def card_hand_transition_class(n_hands: int, seed: int, max_value: int, n_suits: int) -> tuple:
    pair_hits = 0
    suited_hits = 0
    gap_sum = 0
    score_sum = 0

    for i in range(n_hands):
        x = (seed * 1103515245 + i * 12345) & 0x7FFFFFFF
        y = (seed * 214013 + i * 2531011 + score_sum) & 0x7FFFFFFF
        c1 = Card((x % max_value) + 2, x % n_suits)
        c2 = Card((y % max_value) + 2, y % n_suits)
        h = Hand(c1, c2)
        pair_hits += h.is_pair()
        suited_hits += h.is_suited()
        gap_sum += h.gap()
        score_sum = (score_sum + h.score()) & 0xFFFFFFFF

    return (pair_hits, suited_hits, gap_sum, score_sum)
