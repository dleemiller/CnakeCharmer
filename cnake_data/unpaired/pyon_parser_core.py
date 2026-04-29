"""Recursive-descent parser over pre-tokenized stream."""

from __future__ import annotations


def parse_tokens(tokens, to_parse=""):
    i = 0
    n = len(tokens)

    def read_value():
        nonlocal i
        if i >= n:
            raise Exception("out of tokens")
        t = tokens[i]
        i += 1
        typ = t["type"]
        val = t.get("value")
        if typ in ("QUOTED", "NUMBER"):
            return val
        if typ == "BAREWORD":
            if i < n and tokens[i]["type"] == "OPEN_PAREN":
                i += 1
                return read_pyob(val)
            lw = val.lower()
            if lw == "true":
                return True
            if lw == "false":
                return False
            if lw in ("none", "null"):
                return None
            if lw == "nan":
                return float("nan")
            if lw == "inf":
                return float("inf")
            raise Exception("bad bareword:" + val)
        if typ == "OPEN_BRACKET":
            return read_list()
        if typ == "OPEN_CURLY":
            return read_dict()
        raise Exception("unknown token: %r" % t)

    def read_list():
        nonlocal i
        out = []
        while True:
            if i >= n:
                raise Exception("no ]?")
            t = tokens[i]
            if t["type"] == "CLOSE_BRACKET":
                i += 1
                return out
            if t["type"] == "COMMA":
                i += 1
                continue
            out.append(read_value())

    def read_dict():
        nonlocal i
        out = {}
        while True:
            if i >= n:
                raise Exception("no }?")
            t = tokens[i]
            if t["type"] == "CLOSE_CURLY":
                i += 1
                return out
            if t["type"] == "COMMA":
                i += 1
                continue
            k = read_value()
            if tokens[i]["type"] != "COLON":
                raise Exception("bad dict?")
            i += 1
            out[k] = read_value()

    def read_pyob(name):
        nonlocal i
        out = {"name": name, "ordered": [], "keyed": {}}
        while True:
            if i >= n:
                raise Exception("no )?")
            t = tokens[i]
            if t["type"] == "CLOSE_PAREN":
                i += 1
                return out
            if t["type"] == "COMMA":
                i += 1
                continue
            if t["type"] == "BAREWORD" and i + 1 < n and tokens[i + 1]["type"] == "EQ_SIGN":
                key = t["value"]
                i += 2
                out["keyed"][key] = read_value()
            else:
                out["ordered"].append(read_value())

    return read_value()
