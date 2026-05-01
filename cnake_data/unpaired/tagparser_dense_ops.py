"""Tag parsing and explosion utilities for dense key/value encodings."""

from __future__ import annotations

import json


def parse_tags(keys: list[int], vals: list[int], string_table: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    n = min(len(keys), len(vals))
    for i in range(n):
        out[string_table[keys[i]]] = string_table[vals[i]]
    return out


def parse_dense_tags(keys_vals: list[int], string_table: list[str]) -> list[dict[str, str] | None]:
    """Parse [k,v,k,v,0,k,v,0,...] into per-object tag dicts."""
    tag_list: list[dict[str, str] | None] = []
    tag_idx = 0
    n = len(keys_vals)

    while tag_idx < n:
        tags: dict[str, str] = {}
        while tag_idx < n and keys_vals[tag_idx] != 0:
            k = keys_vals[tag_idx]
            v = keys_vals[tag_idx + 1]
            tags[string_table[k]] = string_table[v]
            tag_idx += 2

        tag_list.append(tags if tags else None)
        tag_idx += 1

    return tag_list


def explode_way_tags(ways: list[dict]) -> list[dict]:
    """Inline 'tags' dictionary into top-level fields for each way object."""
    exploded: list[dict] = []
    for way in ways:
        item = dict(way)
        tags = item.pop("tags", {})
        for key, value in tags.items():
            item[key] = value
        exploded.append(item)
    return exploded


def explode_tag_array(
    tag_array: list[dict[str, str]], tags_as_columns: list[str]
) -> dict[str, list[str | None]]:
    lookup = set(tags_as_columns)
    data: dict[str, list[str | None]] = {k: [] for k in tags_as_columns}
    data["tags"] = []

    for tag in tag_array:
        tag_records = {k: None for k in tags_as_columns}
        other_tags: dict[str, str] = {}

        for key, value in tag.items():
            if key in lookup:
                tag_records[key] = value
            else:
                other_tags[key] = value

        for key, value in tag_records.items():
            data[key].append(value)
        data["tags"].append(json.dumps(other_tags) if other_tags else None)

    return data
