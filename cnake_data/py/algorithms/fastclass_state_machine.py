"""Class-based state machine transitions over deterministic events.

Keywords: algorithms, class, state machine, object methods, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class FastClass:
    def __init__(self, state: int, gain: int):
        self.state = state
        self.gain = gain

    def step(self, event: int, mask: int) -> int:
        self.state = (self.state * self.gain + event) & mask
        return self.state


@python_benchmark(args=(64, 250000, 1337, 1023))
def fastclass_state_machine(n_objs: int, steps: int, seed: int, mask: int) -> tuple:
    objs = [FastClass((seed + i * 17) & mask, 3 + (i % 5)) for i in range(n_objs)]
    checksum = 0
    hits = 0

    for t in range(steps):
        idx = t % n_objs
        event = (seed * 1103515245 + t * 12345) & mask
        s = objs[idx].step(event, mask)
        checksum = (checksum + s) & 0xFFFFFFFF
        if (s & 31) == (idx & 31):
            hits += 1

    return (checksum, hits, objs[-1].state)
