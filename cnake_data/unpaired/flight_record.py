"""Flight record data container with equality semantics."""

from __future__ import annotations


class CFlight:
    def __init__(self, city_from, city_to, day, price, int_id=-1):
        self.int_id = int_id
        self.city_from = city_from
        self.city_to = city_to
        self.day = int(day)
        self.price = int(price)

    def __str__(self):
        return f"{self.city_from} {self.city_to} {self.day} {self.price}"

    def __repr__(self):
        return f"<Flight: {self}>"

    def __eq__(self, other):
        return (
            self.city_from == other.city_from
            and self.city_to == other.city_to
            and self.day == other.day
            and self.price == other.price
        )
