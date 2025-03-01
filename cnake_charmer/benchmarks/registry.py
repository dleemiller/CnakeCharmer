"""
Benchmark Registry Module.

This module provides a registry for benchmark items along with decorators
to register implementations. Two decorators are provided: one for registering
a Python implementation and one for registering a Cython implementation.

Example usage:

    from cnake_charmer.benchmarks.registry import python_benchmark, cython_benchmark

    @python_benchmark(args=(10000,))
    def fizzbuzz(n: int) -> List[str]:
        # Python implementation here.
        ...

    @cython_benchmark(args=(10000,))
    def fizzbuzz(n: int) -> List[str]:
        # Cython implementation here.
        ...
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Literal


class Variant(Enum):
    """Enum for benchmark implementation variants."""

    PYTHON = auto()
    CYTHON = auto()
    CYTHON_PP = auto()


@dataclass
class BenchmarkItem:
    """Data class representing a benchmark registration item.

    Attributes:
        benchmark_id (str): Unique identifier for the benchmark.
        variant (Variant): Implementation variant (Python or Cython).
        func (Callable): The benchmark function.
        args (Tuple[Any, ...]): Positional arguments for the benchmark.
        kwargs (Dict[str, Any]): Keyword arguments for the benchmark.
        num_runs (int): Number of times to run the benchmark.
    """

    benchmark_id: str
    variant: Variant
    func: Callable
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    num_runs: int = 100


# Global registry mapping benchmark IDs to a dict of Variant to BenchmarkItem.
benchmark_registry: Dict[str, Dict[Variant, BenchmarkItem]] = {}


def _register_benchmark(
    variant: Variant,
    func: Callable,
    *,
    benchmark_id: Optional[str] = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_runs: int = 100,
) -> Callable:
    """Base decorator to register a benchmark.

    Args:
        variant (Variant): The variant of the benchmark.
        func (Callable): The function being decorated.
        benchmark_id (Optional[str]): Unique benchmark ID. If not provided,
            the function's __name__ is used.
        args (Tuple[Any, ...]): Positional arguments for the benchmark.
        kwargs (Optional[Dict[str, Any]]): Keyword arguments for the benchmark.
        num_runs (int): Number of iterations for the benchmark.

    Returns:
        Callable: The decorated function.
    """
    if kwargs is None:
        kwargs = {}
    if benchmark_id is None:
        benchmark_id = func.__name__

    item = BenchmarkItem(
        benchmark_id=benchmark_id,
        variant=variant,
        func=func,
        args=args,
        kwargs=kwargs,
        num_runs=num_runs,
    )

    if benchmark_id not in benchmark_registry:
        benchmark_registry[benchmark_id] = {}
    benchmark_registry[benchmark_id][variant] = item

    return func


def python_benchmark(
    *,
    benchmark_id: Optional[str] = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_runs: int = 100,
) -> Callable[[Callable], Callable]:
    """Decorator to register a Python benchmark implementation.

    Args:
        benchmark_id (Optional[str]): Unique identifier for the benchmark.
            If not provided, the decorated function's __name__ is used.
        args (Tuple[Any, ...]): Positional arguments for the benchmark.
        kwargs (Optional[Dict[str, Any]]): Keyword arguments for the benchmark.
        num_runs (int): Number of iterations to run the benchmark.

    Returns:
        Callable: A decorator that registers the function as a Python benchmark.
    """
    return partial(
        _register_benchmark,
        Variant.PYTHON,
        benchmark_id=benchmark_id,
        args=args,
        kwargs=kwargs,
        num_runs=num_runs,
    )


def cython_benchmark(
    syntax=Literal["cy", "pp"],
    *,
    benchmark_id: Optional[str] = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_runs: int = 100,
) -> Callable[[Callable], Callable]:
    """Decorator to register a Cython benchmark implementation.

    Args:
        benchmark_id (Optional[str]): Unique identifier for the benchmark.
            If not provided, the decorated function's __name__ is used.
        args (Tuple[Any, ...]): Positional arguments for the benchmark.
        kwargs (Optional[Dict[str, Any]]): Keyword arguments for the benchmark.
        num_runs (int): Number of iterations to run the benchmark.

    Returns:
        Callable: A decorator that registers the function as a Cython benchmark.
    """
    assert syntax in ["cy", "pp"]
    return partial(
        _register_benchmark,
        Variant.CYTHON if syntax == "cy" else Variant.CYTHON_PP,
        benchmark_id=benchmark_id,
        args=args,
        kwargs=kwargs,
        num_runs=num_runs,
    )
