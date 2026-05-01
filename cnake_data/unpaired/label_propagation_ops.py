"""Label propagation and span-group reduce/expand utilities."""

from __future__ import annotations


def get_most_common_label(labels, output_size=1, not_entity=0):
    if not labels:
        return [not_entity] * output_size
    counter = {}
    for i in labels:
        counter[i] = counter.get(i, 0) + 1
    best = max(counter.items(), key=lambda kv: kv[1])[0]
    return [best] * output_size


def compute_output_length(input_seq, replacement_group):
    offset = 0
    for rep in replacement_group:
        after = len(rep["new_value"])
        before = rep["end"] - rep["start"]
        offset += after - before
    return len(input_seq) + offset


def propagate_by_replacement_group(labels, replacement_group, transduce_func=None, not_entity=0):
    if transduce_func is None:
        transduce_func = lambda x, n: get_most_common_label(x, n, not_entity)
    out_len = compute_output_length(labels, replacement_group)
    out = [not_entity] * out_len

    i_start = o_start = 0
    for rep in replacement_group:
        fixed_len = rep["start"] - i_start
        out[o_start : o_start + fixed_len] = labels[i_start : rep["start"]]
        o_start += fixed_len
        expand = len(rep["new_value"])
        out[o_start : o_start + expand] = transduce_func(labels[rep["start"] : rep["end"]], expand)
        i_start = rep["end"]
        o_start += expand
    out[o_start:] = labels[i_start:]
    return out
