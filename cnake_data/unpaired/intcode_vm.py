def get_param(codes, index, mode):
    return codes[codes[index]] if mode == 0 else codes[index]


def run(program_csv, inp=5, mem_size=1000):
    """Run Intcode-like VM and return last output."""
    codes = [0] * mem_size
    for i, x in enumerate(program_csv.split(",")):
        codes[i] = int(x)

    out = 0
    i = 0

    while True:
        op = codes[i] % 100
        m0 = (codes[i] // 100) % 10
        m1 = (codes[i] // 1000) % 10

        if op == 99:
            break

        if op == 1:
            codes[codes[i + 3]] = get_param(codes, i + 1, m0) + get_param(codes, i + 2, m1)
            i += 4
        elif op == 2:
            codes[codes[i + 3]] = get_param(codes, i + 1, m0) * get_param(codes, i + 2, m1)
            i += 4
        elif op == 3:
            codes[codes[i + 1]] = inp
            i += 2
        elif op == 4:
            out = get_param(codes, i + 1, m0)
            i += 2
        elif op == 5:
            if get_param(codes, i + 1, m0) != 0:
                i = get_param(codes, i + 2, m1)
            else:
                i += 3
        elif op == 6:
            if get_param(codes, i + 1, m0) == 0:
                i = get_param(codes, i + 2, m1)
            else:
                i += 3
        elif op == 7:
            codes[codes[i + 3]] = (
                1 if get_param(codes, i + 1, m0) < get_param(codes, i + 2, m1) else 0
            )
            i += 4
        elif op == 8:
            codes[codes[i + 3]] = (
                1 if get_param(codes, i + 1, m0) == get_param(codes, i + 2, m1) else 0
            )
            i += 4
        else:
            raise ValueError(f"Unknown opcode {op} at {i}")

    return out
