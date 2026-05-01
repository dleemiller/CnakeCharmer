"""Variable-vector code parsing and construction helpers."""

from __future__ import annotations

import re


class BaseNode:
    def is_rv(self):
        return True


class DiscreteRV(BaseNode):
    def __init__(self, num_values, name=""):
        self.num_values = int(num_values)
        self.name = name
        self.value = None

    def is_discrete(self):
        return True

    def value_in_support(self, val):
        if self.num_values == -1:
            return int(val) >= 0
        if self.num_values == -2:
            return float(val).is_integer()
        return float(val).is_integer() and 0 <= int(val) < self.num_values


class ContinuousRV(BaseNode):
    def __init__(self, support=None, name=""):
        self.support = support
        self.name = name
        self.value = None

    def is_discrete(self):
        return False

    def value_in_support(self, val):
        if self.support is None:
            return True
        x = float(val)
        return any(lo <= x <= hi for lo, hi in self.support)


class VariableVector(list):
    def add_variable(self, var):
        if not isinstance(var, BaseNode) or var in self:
            raise ValueError("cannot add non-rv or duplicate rv")
        self.append(var)

    def set(self, values, check_support=True):
        if len(values) != len(self):
            raise ValueError("values length mismatch")
        for var, val in zip(self, values, strict=False):
            if check_support and not var.value_in_support(val):
                raise ValueError("value not in support")
            var.value = float(val)

    def num_instantiations(self):
        total = 1
        for var in self:
            total *= getattr(var, "num_values", 1)
        return total


def tuple_code(spec):
    mo = re.match(
        r"(?i)(?P<num>\\d*)(?P<type>[dc])(?P<spec>\\d+|-\\d+|(\\(\\s*(\\([^)]+\\)[^(]*)+\\s*\\))*)(?P<name>.*$)",
        spec,
    )
    if not mo:
        raise ValueError("invalid code string")
    num = 1 if mo.group("num") == "" else int(mo.group("num"))
    tp = mo.group("type").lower()
    if tp == "d":
        parsed_spec = 2 if mo.group("spec") == "" else int(mo.group("spec"))
    else:
        if mo.group("spec") == "":
            parsed_spec = None
        else:
            vals = [float(x) for x in re.split(r"[(),\\s]", mo.group("spec")) if x != ""]
            parsed_spec = tuple((vals[i], vals[i + 1]) for i in range(0, len(vals), 2))
    return (num, tp, parsed_spec, mo.group("name"))


def vvec(code):
    if hasattr(code, "flat"):
        codes = [(1, "d", int(c), "") if int(c) != 0 else (1, "c", None, "") for c in code.flat]
    elif isinstance(code, str):
        codes = [tuple_code(c) for c in code.split("*")]
    elif hasattr(code, "__iter__"):
        codes = [c if isinstance(c, tuple) else tuple_code(c) for c in code]
    else:
        raise ValueError("invalid code type")

    out = VariableVector()
    for item in codes:
        if isinstance(item, BaseNode):
            out.add_variable(item)
            continue
        for _ in range(item[0]):
            rv = DiscreteRV(item[2], item[3]) if item[1] == "d" else ContinuousRV(item[2], item[3])
            out.add_variable(rv)
    return out
