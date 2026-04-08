"""
Atomic wiki page writes with page-level file locking.

Prevents data loss when multiple processes (e.g. parallel reflection agents)
write to the same wiki page concurrently.
"""

import fcntl
from pathlib import Path


def atomic_wiki_write(page_path: Path, content: str) -> None:
    """Write wiki page content with page-level file lock.

    Uses fcntl.flock for advisory locking and atomic rename for crash safety.

    Args:
        page_path: Path to the .md wiki page file.
        content: Full markdown content to write.
    """
    page_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = page_path.with_suffix(".md.lock")

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            tmp = page_path.with_suffix(".md.tmp")
            tmp.write_text(content)
            tmp.rename(page_path)  # atomic on same filesystem
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
