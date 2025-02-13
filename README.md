# CnakeCharmer

A "living dataset" of python -> cython pair implementations


## Project Goals

Due to the vast amount of python code available for training language models,
LLMs are increasingly getting better at producing python code.
Due to the similarity in syntax between Python and Cython, LLMs are likely capable of making translations
between Python and Cyth    logging.info(benchmark_registry.items())
on code. This is a desirable capability, as it would facilitate tools that can
profile and automate performance improving-capabilities to existing codebases.

Currently however, automated translation capability seems to underperform, with simple syntax mistakes that
could be easily improved with example data.

To that end, we set a target of **10,000 Python->Cython implementations** as a "living codebase" that can be
compiled, benchmarked and tested. This allows us to write the dataset as code, while testing performance,
correctness and making implementation changes that can progress toward a gold standard benchmark.

## Project Structure

The dataset goal is to create mirror implementations between python and cython. These implementations go into
modules mirroring one another at `cnake_charmer/(py || cy)/...`. Due to the size, and potential usefulness of a
taxonomy, it should be organizing implementations into an appropriately submodule.

### Testing

Testing should be done for correctness. More importantly, it should be done to check that the implementations have
equivalent inputs and outputs.

### Benchmarking

There are many ways to optimizing both python and cython code. Therefore, we'd like to ensure that implementation
optimizations do not regress in performance.

The benchmarking decorators should be applied:
```python
from cnake_charmer.benchmarks import python_benchmark
from typing import List

@python_benchmark(args=(10000,))
def fizzbuzz(n: int) -> List[str]:
```

Note that the decorators use the `__name__` property. For matching purposes, this means the script name should be
globally unique, but mirrored across the py/cy implementation mirror.


## External Libraries

"Python" implementations should be "pure python". Where it makes sense that a library should be added to the project,
we should ask:

- is this a library that would commonly benefit from being used in *cython* code?
- can our script be reasonably implemented without it?
- will it be useful for multiple different code implementations?


## Contributing

*Please do not contribute to the project at this point.* Our first priority will be to setup CI tools that
can enable us to automatically benchmark and test PRs. Once we are settled on the bones of the project,
we will open it up for contributions.
