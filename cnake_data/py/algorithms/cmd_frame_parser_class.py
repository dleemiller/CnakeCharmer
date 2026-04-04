"""Class-based framed byte parser with payload aggregation.

Keywords: algorithms, class, parser, bytes, stateful object, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class FrameParser:
    def __init__(self, payload_len: int):
        self.payload_len = payload_len
        self.pos = 0
        self.running = 0

    def feed_byte(self, b: int, frame_sum: int) -> tuple[int, int]:
        self.running = (self.running + b + self.pos) & 0xFFFFFFFF
        self.pos += 1
        if self.pos == self.payload_len:
            out = (frame_sum ^ self.running) & 0xFFFFFFFF
            self.pos = 0
            self.running = 0
            return (1, out)
        return (0, 0)


@python_benchmark(args=(7000, 31, 17, 1021))
def cmd_frame_parser_class(n_frames: int, payload_len: int, seed: int, mod: int) -> tuple:
    parser = FrameParser(payload_len)
    checksum = 0
    frame_total = 0
    max_frame = 0

    for i in range(n_frames):
        frame_sum = 0
        for j in range(payload_len):
            b = (seed * 1315423911 + i * 31337 + j * 97 + frame_total) & mod
            frame_sum = (frame_sum + b + j) & 0xFFFFFFFF
            done, out = parser.feed_byte(b, frame_sum)
            if done:
                frame_total += 1
                checksum = (checksum + out) & 0xFFFFFFFF
                if out > max_frame:
                    max_frame = out

    return (frame_total, checksum, max_frame)
