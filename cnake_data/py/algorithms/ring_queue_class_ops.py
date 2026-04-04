"""Class-based ring queue push/pop workload metrics.

Keywords: algorithms, class, queue, ring buffer, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Queue:
    def __init__(self, capacity: int):
        self.buf = [0] * capacity
        self.cap = capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def push(self, x: int) -> None:
        if self.size == self.cap:
            self.head = (self.head + 1) % self.cap
            self.size -= 1
        self.buf[self.tail] = x
        self.tail = (self.tail + 1) % self.cap
        self.size += 1

    def pop(self) -> int:
        if self.size == 0:
            return -1
        x = self.buf[self.head]
        self.head = (self.head + 1) % self.cap
        self.size -= 1
        return x


@python_benchmark(args=(512, 900000, 17))
def ring_queue_class_ops(capacity: int, rounds: int, seed: int) -> tuple:
    q = Queue(capacity)
    checksum = 0
    popped = 0
    for i in range(rounds):
        x = (seed * 1664525 + i * 1013904223) & 0xFFFFFFFF
        q.push(x)
        if i & 1:
            v = q.pop()
            if v >= 0:
                checksum = (checksum + (v & 0xFFFF)) & 0xFFFFFFFF
                popped += 1
    return (checksum, popped, q.size)
