"""Recursive file watcher and round-robin multiprocess scheduler."""

from __future__ import annotations

import re
from enum import Enum
from multiprocessing import Pipe, Process
from os import scandir


class Change(Enum):
    added = "added"
    modified = "modified"
    deleted = "deleted"


def or_pattern(patterns: list[str]):
    return re.compile("|".join(f"(?:{p})" for p in patterns)).match


class AllWatcher:
    def __init__(self, root_path: str):
        self.files: dict[str, float] = {}
        self.root_path = root_path
        self.check()

    def should_watch_dir(self, entry) -> bool:
        return True

    def should_watch_file(self, entry) -> bool:
        return True

    def _walk(self, dir_path: str, changes: set, new_files: dict[str, float]) -> None:
        for entry in scandir(dir_path):
            if entry.is_dir():
                if self.should_watch_dir(entry):
                    self._walk(entry.path, changes, new_files)
            elif self.should_watch_file(entry):
                mtime = entry.stat().st_mtime
                new_files[entry.path] = mtime
                old = self.files.get(entry.path)
                if old is None:
                    changes.add((Change.added, entry.path))
                elif old != mtime:
                    changes.add((Change.modified, entry.path))

    def check(self) -> set:
        changes = set()
        new_files: dict[str, float] = {}
        try:
            self._walk(self.root_path, changes, new_files)
        except OSError:
            pass
        deleted = self.files.keys() - new_files.keys()
        for entry in deleted:
            changes.add((Change.deleted, entry))
        self.files = new_files
        return changes


class SearchPathsWatcher(AllWatcher):
    IGNORED_DIRS = {".git", "__pycache__", "site-packages", "env", "venv", ".env", ".venv"}

    def should_watch_dir(self, entry) -> bool:
        return entry.name not in self.IGNORED_DIRS

    def should_watch_file(self, entry) -> bool:
        return entry.name.endswith(".py")


class MPScheduler:
    def __init__(self, num_processes: int):
        self.num_processes = num_processes
        self._pipe_snd = []
        self._pipe_rcv = []
        self._procs = []
        self._next_pipe_ndx = 0
        for _ in range(self.num_processes):
            snd, rcv = Pipe()
            self._pipe_snd.append(snd)
            self._pipe_rcv.append(rcv)

    def _target(self, worker_id: str, jobs) -> None:
        while True:
            item = jobs.recv()
            if item == "<stop>":
                break

    def _spawn_procs(self) -> None:
        assert self._procs == []
        for n in range(self.num_processes):
            p = Process(target=self._target, args=(f"worker-{n}", self._pipe_rcv[n]), daemon=True)
            self._procs.append(p)
            p.start()

    def _kill_procs(self) -> None:
        for p in self._pipe_snd:
            try:
                p.send("<stop>")
            except Exception:
                pass
        for p in self._procs:
            try:
                p.join()
            except Exception:
                pass
        self._procs = []

    def close(self) -> None:
        self._kill_procs()
        for p in self._pipe_rcv + self._pipe_snd:
            try:
                p.close()
            except Exception:
                pass

    def __enter__(self):
        self._spawn_procs()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._kill_procs()

    def submit_one(self, item) -> None:
        idx = self._next_pipe_ndx
        self._pipe_snd[idx].send(item)
        idx += 1
        self._next_pipe_ndx = 0 if idx == self.num_processes else idx
