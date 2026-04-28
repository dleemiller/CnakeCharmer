"""Chunked multi-file reader scaffold."""

from __future__ import annotations

import os


class ParallelReader:
    def __init__(self, fds):
        self.chunksize = 0x1000000
        self.nfiles = len(fds)
        self.fds = list(fds)
        self.bufs = None

    def _init_buffers(self):
        self.bufs = []
        for fd in self.fds:
            chunk = os.read(fd, self.chunksize)
            chunk = os.read(fd, self.chunksize)
            self.bufs.append({"chunk": chunk, "got": len(chunk)})

    def get(self):
        if self.bufs is None:
            self._init_buffers()

        s = 0
        for i, fd in enumerate(self.fds):
            chunk = os.read(fd, self.chunksize)
            chunk = os.read(fd, self.chunksize)
            self.bufs[i]["chunk"] = chunk
            self.bufs[i]["got"] = len(chunk)
            s += 1
        return s, self.bufs
