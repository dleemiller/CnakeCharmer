"""
Base types for code sources.

A ProblemSpec represents a single Python/Cython pair (or partial pair)
that can be validated, benchmarked, and used for training.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ProblemSpec:
    problem_id: str
    description: str = ""
    python_code: str = ""
    cython_code: str = ""
    func_name: str = ""
    test_cases: list = field(default_factory=list)
    benchmark_args: tuple | None = None
    category: str = ""  # e.g. "numerical", "string", "array", "algorithms"
    difficulty: str = ""  # "easy", "medium", "hard"
    source: str = ""  # e.g. "stack_v2", "algorithmic", "synthetic", "manual"
    metadata: dict = field(default_factory=dict)

    @property
    def has_python(self) -> bool:
        return bool(self.python_code.strip())

    @property
    def has_cython(self) -> bool:
        return bool(self.cython_code.strip())

    @property
    def is_complete(self) -> bool:
        return self.has_python and self.has_cython and bool(self.func_name)


class CodeSource(Protocol):
    """Protocol for code sources that yield ProblemSpecs."""

    def yield_problems(self) -> Iterator[ProblemSpec]: ...
