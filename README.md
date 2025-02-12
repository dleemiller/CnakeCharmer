# CnakeCharmer

A "living dataset" of python -> cython pair implementations


## Project Goals

Due to the vast amount of python code available for training language models,
LLMs are increasingly getting better at producing python code.
Due to the similarity in syntax between Python and Cython, LLMs are likely capable of making translations
between Python and Cython code. This is a desirable capability, as it would facilitate tools that can
profile and automate performance improving-capabilities to existing codebases.

Currently however, automated translation capability seems to underperform, with simple syntax mistakes that
could be easily improved with example data.

To that end, we set a target of **10,000 Python->Cython implementations** as a "living codebase" that can be
compiled, benchmarked and tested. This allows us to write the dataset as code, while testing performance,
correctness and making implementation changes that can progress toward a gold standard benchmark.


## Contributing

*Please do not contribute to the project at this point.* Our first priority will be to setup CI tools that
can enable us to automatically benchmark and test PRs. Once we are settled on the bones of the project,
we will open it up for contributions.
