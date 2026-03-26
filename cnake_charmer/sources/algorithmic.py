"""
Source: Curated algorithmic problems from a JSONL catalog.

Loads problems from data/problems.jsonl with Python reference implementations.
These have python_code filled and need cython_code to be generated.
"""

import json
import logging
import os
from collections.abc import Iterator

from cnake_charmer.sources.base import ProblemSpec

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "problems.jsonl",
)


class AlgorithmicSource:
    """
    Yield ProblemSpecs from a curated JSONL catalog of algorithmic problems.

    Each line in the JSONL file should have:
    {
        "problem_id": "algo_001",
        "description": "Compute the Nth Fibonacci number",
        "python_code": "def fibonacci(n): ...",
        "func_name": "fibonacci",
        "test_cases": [[[10], {}], [[20], {}]],
        "benchmark_args": [30],
        "category": "numerical",
        "difficulty": "easy"
    }
    """

    def __init__(
        self,
        catalog_path: str = DEFAULT_CATALOG_PATH,
        categories: list | None = None,
        difficulties: list | None = None,
        limit: int | None = None,
    ):
        self.catalog_path = catalog_path
        self.categories = categories
        self.difficulties = difficulties
        self.limit = limit

    def yield_problems(self) -> Iterator[ProblemSpec]:
        if not os.path.exists(self.catalog_path):
            logger.warning(f"Catalog not found at {self.catalog_path}")
            return

        count = 0
        with open(self.catalog_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line: {e}")
                    continue

                # Filter by category/difficulty
                if self.categories and data.get("category") not in self.categories:
                    continue
                if self.difficulties and data.get("difficulty") not in self.difficulties:
                    continue

                benchmark_args = data.get("benchmark_args")
                if benchmark_args is not None:
                    benchmark_args = tuple(benchmark_args)

                yield ProblemSpec(
                    problem_id=data.get("problem_id", f"algo_{count}"),
                    description=data.get("description", ""),
                    python_code=data.get("python_code", ""),
                    cython_code=data.get("cython_code", ""),
                    func_name=data.get("func_name", ""),
                    test_cases=data.get("test_cases", []),
                    benchmark_args=benchmark_args,
                    category=data.get("category", ""),
                    difficulty=data.get("difficulty", ""),
                    source="algorithmic",
                    metadata=data.get("metadata", {}),
                )

                count += 1
                if self.limit and count >= self.limit:
                    return
