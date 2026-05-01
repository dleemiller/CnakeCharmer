"""Pattern extraction over text windows with recursive token constraints."""

from __future__ import annotations

import re


def find_len(query: list[str]) -> int:
    length = 0
    curs = 0
    while curs < len(query) - 1:
        length += int(query[curs + 2]) + len(query[curs + 3])
        curs += 3
    return length


def search_word(query: list[str], text_found: str, index: int, matches: list[int]) -> list[int]:
    for word in re.finditer(query[3], text_found):
        if word.start() <= int(query[2]) - int(query[1]):
            if len(query) == 4:
                matches.append(index + word.end())
            else:
                offset = word.end() + int(query[4])
                search_word(query[3:], text_found[offset:], index + offset, matches)
    return matches


def create_query(args: list[str]) -> list[str]:
    query: list[str] = []
    for arg in args[1:]:
        for elem in re.split(r"\[|,|\]", arg):
            if elem:
                query.append(elem)
    return query


def extract_patterns(lines: list[str], query: list[str]) -> list[str]:
    out: list[str] = []
    size = find_len(query)
    for line in lines:
        for word in re.finditer(query[0], line):
            start = word.end() + int(query[1])
            window = line[start : word.end() + size]
            result = search_word(query, window, start, [])
            for end_idx in result:
                chunk = line[word.start() : end_idx]
                if "\n" not in chunk:
                    out.append(chunk)
    return out
