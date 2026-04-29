"""Poker hand score comparison and best-hand search."""

from __future__ import annotations

import itertools

NO_SCORE = -1
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8
HAND_LENGTH = 5


class HandScore:
    def __init__(self, type_=NO_SCORE):
        self.type = type_
        self.kicker = None

    def __repr__(self):
        return f"{self.type}, {self.kicker}"

    def _cmp_tuple(self):
        return (self.type, self.kicker)

    def __lt__(self, other):
        return self._cmp_tuple() < other._cmp_tuple()


class HandBuilder:
    def __init__(self, cards):
        self.cards = list(cards)

    def find_hand(self):
        if not self.cards or len(self.cards) < HAND_LENGTH:
            return None, None

        best_hand_score = HandScore()
        best_hand = None
        for hand in itertools.combinations(self.cards, HAND_LENGTH):
            score = HandBuilder(list(hand)).score_hand()
            if best_hand_score < score:
                best_hand_score = score
                best_hand = hand
        return best_hand, best_hand_score

    def score_hand(self):
        score = HandScore()
        if not self.cards or len(self.cards) != HAND_LENGTH:
            return score

        seen = [0] * 16
        for card in self.cards:
            seen[card.value] += 1

        counts = sorted([(cnt, val) for val, cnt in enumerate(seen) if cnt > 0], reverse=True)
        vals_desc = sorted((card.value for card in self.cards), reverse=True)

        if counts[0][0] == 4:
            score.type = QUADS
            score.kicker = tuple([counts[0][1], counts[1][1]])
        elif counts[0][0] == 3 and counts[1][0] == 2:
            score.type = FULL_HOUSE
            score.kicker = tuple([counts[0][1], counts[1][1]])
        elif counts[0][0] == 3:
            score.type = TRIPS
            score.kicker = tuple([counts[0][1]] + [v for v in vals_desc if v != counts[0][1]])
        elif counts[0][0] == 2 and counts[1][0] == 2:
            score.type = TWO_PAIR
            pairs = sorted([counts[0][1], counts[1][1]], reverse=True)
            kicker = [v for v in vals_desc if v not in pairs][0]
            score.kicker = tuple(pairs + [kicker])
        elif counts[0][0] == 2:
            score.type = PAIR
            p = counts[0][1]
            score.kicker = tuple([p] + [v for v in vals_desc if v != p])
        else:
            score.type = HIGH_CARD
            score.kicker = tuple(vals_desc)
        return score
