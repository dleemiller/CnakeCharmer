"""
Source: Cython code from The Stack v2 dataset.

Reads Cython source files from the local DuckDB database.
These have cython_code filled but need python_code to be generated.
"""

import logging
import os
from collections.abc import Iterator

import duckdb

from cnake_charmer.sources.base import ProblemSpec

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "utils",
    "stack_data",
    "stack_cython_1k.duckdb",
)


class StackV2Source:
    """
    Yield ProblemSpecs from The Stack v2 Cython data.

    Each problem has cython_code filled from the dataset.
    python_code needs to be generated (reverse direction).
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        min_length: int = 100,
        max_length: int = 10000,
        exclude_generated: bool = True,
        limit: int | None = None,
    ):
        self.db_path = db_path
        self.min_length = min_length
        self.max_length = max_length
        self.exclude_generated = exclude_generated
        self.limit = limit

    def yield_problems(self) -> Iterator[ProblemSpec]:
        if not os.path.exists(self.db_path):
            logger.warning(f"DuckDB not found at {self.db_path}")
            return

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            query = """
                SELECT blob_id, path, repo_name, filename, content, length_bytes
                FROM stack_cython
                WHERE content IS NOT NULL
                  AND length_bytes >= ?
                  AND length_bytes <= ?
            """
            params = [self.min_length, self.max_length]

            if self.exclude_generated:
                query += " AND is_generated = false"

            query += " ORDER BY length_bytes ASC"

            if self.limit:
                query += f" LIMIT {self.limit}"

            rows = con.execute(query, params).fetchall()

            for row in rows:
                blob_id, path, repo_name, filename, content, length_bytes = row

                yield ProblemSpec(
                    problem_id=f"stack_v2_{blob_id[:12]}",
                    description=f"Cython code from {repo_name}: {path}",
                    cython_code=content,
                    func_name="",  # Needs extraction
                    category=_guess_category(path, content),
                    source="stack_v2",
                    metadata={
                        "blob_id": blob_id,
                        "repo_name": repo_name,
                        "path": path,
                        "filename": filename,
                        "length_bytes": length_bytes,
                    },
                )
        finally:
            con.close()


def _guess_category(path: str, content: str) -> str:
    """Rough category guess from file path and content."""
    path_lower = path.lower()
    content_lower = content[:500].lower()

    if any(kw in path_lower for kw in ["math", "linalg", "algebra", "matrix"]):
        return "numerical"
    if any(kw in path_lower for kw in ["sort", "search", "algorithm"]):
        return "algorithms"
    if any(kw in path_lower for kw in ["string", "text", "parse"]):
        return "string_processing"
    if any(kw in path_lower for kw in ["image", "vision", "pixel"]):
        return "image_processing"
    if any(kw in content_lower for kw in ["numpy", "ndarray", "memoryview"]):
        return "numerical"
    if any(kw in content_lower for kw in ["cdef class", "extension type"]):
        return "extension_types"

    return "general"
