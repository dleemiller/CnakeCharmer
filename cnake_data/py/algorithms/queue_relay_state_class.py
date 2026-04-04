"""Class-based queue relay workload with push/pop transitions.

Keywords: algorithms, class, queue, relay, stateful object, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class RelayQueue:
    def __init__(self, capacity: int):
        self.buf = [0] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.capacity = capacity

    def push(self, value: int) -> None:
        if self.size == self.capacity:
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
        self.buf[self.tail] = value
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1

    def pop(self) -> int:
        if self.size == 0:
            return -1
        out = self.buf[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return out


@python_benchmark(args=(1024, 320000, 1337, 4095))
def queue_relay_state_class(capacity: int, rounds: int, seed: int, mask: int) -> tuple:
    q = RelayQueue(capacity)
    checksum = 0
    hits = 0
    last = 0

    for t in range(rounds):
        x = (seed * 1103515245 + t * 12345 + checksum) & mask
        q.push(x)
        if (t & 3) != 0:
            y = q.pop()
            if y >= 0:
                last = (y ^ t) & mask
                checksum = (checksum + last + q.size) & 0xFFFFFFFF
                if (last & 63) == (t & 63):
                    hits += 1

    return (checksum, hits, q.size, last)
