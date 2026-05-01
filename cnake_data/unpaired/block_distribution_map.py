"""Block ownership/local-global index mapping for distributed arrays."""

from __future__ import annotations


class BlockMap:
    def __init__(self, nglobal, nprocs):
        self.nglobal = nglobal
        self.nprocs = nprocs
        self.nlocal = self.nglobal // self.nprocs
        if self.nglobal % self.nprocs > 0:
            self.nlocal += 1

    def owner(self, global_index):
        return global_index // self.nlocal

    def local_index(self, global_index):
        return global_index % self.nprocs

    def global_index(self, owner, local_index):
        return owner * self.nlocal + local_index
