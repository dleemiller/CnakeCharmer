# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based framed byte parser with payload aggregation (Cython)."""

from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class FrameParser:
    cdef int payload_len
    cdef int pos
    cdef unsigned int running

    def __cinit__(self, int payload_len):
        self.payload_len = payload_len
        self.pos = 0
        self.running = 0

    cdef int feed_byte(self, int b, unsigned int frame_sum, unsigned int* out) noexcept nogil:
        self.running = (self.running + <unsigned int>b + <unsigned int>self.pos) & MASK32
        self.pos += 1
        if self.pos == self.payload_len:
            out[0] = (frame_sum ^ self.running) & MASK32
            self.pos = 0
            self.running = 0
            return 1
        return 0


@cython_benchmark(syntax="cy", args=(7000, 31, 17, 1021))
def cmd_frame_parser_class(int n_frames, int payload_len, int seed, int mod):
    cdef FrameParser parser = FrameParser(payload_len)
    cdef int i, j
    cdef unsigned int checksum = 0
    cdef int frame_total = 0
    cdef unsigned int max_frame = 0
    cdef unsigned int frame_sum
    cdef int b
    cdef unsigned int out = 0

    for i in range(n_frames):
        frame_sum = 0
        for j in range(payload_len):
            b = (seed * 1315423911 + i * 31337 + j * 97 + frame_total) & mod
            frame_sum = (frame_sum + <unsigned int>b + <unsigned int>j) & MASK32
            if parser.feed_byte(b, frame_sum, &out):
                frame_total += 1
                checksum = (checksum + out) & MASK32
                if out > max_frame:
                    max_frame = out

    return (frame_total, checksum, max_frame)
