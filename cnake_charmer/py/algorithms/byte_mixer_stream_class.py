"""Class-based byte stream mixer with rolling key schedule.

Keywords: algorithms, class, byte mixer, stream transform, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class ByteMixer:
    def __init__(self, window: int, seed: int, key_scale: int):
        self.window = window
        self.key = [0] * window
        self.pos = 0
        x = seed & 0x7FFFFFFF
        for i in range(window):
            x = (1103515245 * x + 12345 + i * key_scale) & 0x7FFFFFFF
            self.key[i] = x & 255

    def mix_byte(self, b: int, salt: int) -> int:
        k = self.key[self.pos]
        out = (b ^ k ^ (salt & 255)) & 255
        self.key[self.pos] = (k + out + salt + self.pos) & 255
        self.pos += 1
        if self.pos == self.window:
            self.pos = 0
        return out


@python_benchmark(args=(1400000, 97, 5, 29, 11))
def byte_mixer_stream_class(
    n_bytes: int, window: int, rounds: int, seed: int, key_scale: int
) -> tuple:
    mixer = ByteMixer(window, seed, key_scale)
    checksum = 0
    high = 0

    for r in range(rounds):
        salt = seed + r * 101
        for i in range(n_bytes // rounds):
            b = (seed * 53 + i * 17 + r * 19 + checksum) & 255
            out = mixer.mix_byte(b, salt + i)
            checksum = (checksum + out + i) & 0xFFFFFFFF
            if out >= 192:
                high += 1

    key_tail = 0
    for i in range(window):
        key_tail = (key_tail + mixer.key[i] * (i + 1)) & 0xFFFFFFFF

    return (checksum, high, key_tail)
