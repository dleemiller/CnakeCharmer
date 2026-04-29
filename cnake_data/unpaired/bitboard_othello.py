"""Bitboard move generation and flips for Othello/Reversi."""

from __future__ import annotations


def search_offset_left(own, enemy, mask, offset):
    e = enemy & mask
    blank = ~(own | enemy)
    t = e & (own >> offset)
    for _ in range(5):
        t |= e & (t >> offset)
    return blank & (t >> offset)


def search_offset_right(own, enemy, mask, offset):
    e = enemy & mask
    blank = ~(own | enemy)
    t = e & (own << offset)
    for _ in range(5):
        t |= e & (t << offset)
    return blank & (t << offset)


def find_correct_moves(own, enemy):
    lrm = 0x7E7E7E7E7E7E7E7E
    tbm = 0x00FFFFFFFFFFFF00
    m = lrm & tbm
    mobility = 0
    mobility |= search_offset_left(own, enemy, lrm, 1)
    mobility |= search_offset_left(own, enemy, m, 9)
    mobility |= search_offset_left(own, enemy, tbm, 8)
    mobility |= search_offset_left(own, enemy, m, 7)
    mobility |= search_offset_right(own, enemy, lrm, 1)
    mobility |= search_offset_right(own, enemy, m, 9)
    mobility |= search_offset_right(own, enemy, tbm, 8)
    mobility |= search_offset_right(own, enemy, m, 7)
    return mobility


def bit_count(x):
    c = 0
    for _ in range(64):
        c += x & 1
        x >>= 1
    return c
