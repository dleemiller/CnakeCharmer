"""Dance instruction parser and executor (AoC-style)."""

from __future__ import annotations


def instrs(inp: str):
    instructions = []
    for instruction in inp.strip().split(","):
        if instruction[0] == "s":
            d = int(instruction[1:])
            instructions.append(("s", d))
        elif instruction[0] == "x":
            a, b = map(int, instruction[1:].split("/"))
            instructions.append(("x", a, b))
        else:
            n1, n2 = map(lambda x: ord(x) - 97, instruction[1:].split("/"))
            instructions.append(("p", n1, n2))
    return instructions


def go(inp: str, cycles: int = 1):
    progs = list("abcdefghijklmnop")
    instructions = instrs(inp)
    for _ in range(cycles):
        for inst in instructions:
            if inst[0] == "s":
                d = inst[1]
                progs = progs[-d:] + progs[:-d]
            elif inst[0] == "x":
                a, b = inst[1], inst[2]
                progs[a], progs[b] = progs[b], progs[a]
            else:
                a = chr(inst[1] + 97)
                b = chr(inst[2] + 97)
                i1 = progs.index(a)
                i2 = progs.index(b)
                progs[i1], progs[i2] = progs[i2], progs[i1]
    return "".join(progs)
