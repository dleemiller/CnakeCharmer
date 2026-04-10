---
pretty_name: CnakeCharmer Trace Dataset
license: mit
language:
- en
task_categories:
- text-generation
size_categories:
- 100K<n<1M
configs:
- config_name: raw
  data_files:
  - split: train
    path: raw/raw_traces.jsonl
- config_name: parallel
  data_files:
  - split: train
    path: parallel/parallel_examples.jsonl
---

# CnakeCharmer

This dataset is for training an agent to translate python code to cython code, utilizing sandboxed tools for automated testing and optimization.

## Dataset Splits

### Parallel

Parallel python / cython implementations sourced from our [github repo](https://github.com/dleemiller/CnakeCharmer).
We have implemented, compiled, linted, benchmarked and tested over 700 examples.
We use this dataset for collecting agent raw agent traces and GEPA optimizing prompt instructions.

### Raw

Unfiltered trace logs directly from collection runs using a variety of models.
Most traces only use the `evaluate_cython` tool, but a small number of traces also used some experimental `wiki_read` tools.
A small number of these examples are included in the training data, as a natural way of adding some instructional material.
The wiki pages are meant to address real challenges LLMs exhibited while translating code to cython, that we observed in the traces.

### SFT

(planned)

Filtered and formatted data using harmony format for training `gpt-oss`.

### GRPO

(planned)

Unpaired python scripts for multi-step agent training.


## Data Format

Files are stored as JSONL. Each line is one record.

## Tools And Sandbox

The trace and pair evaluation flow is built around the `evaluate_cython` tool:

- Compiles Cython code
- Runs correctness checks against Python reference behavior
- Runs benchmark comparisons (Python vs Cython) to compute speedup
- Produces optimization/quality metrics used for filtering and training

Execution is sandboxed for safety and stability:

- Runtime code execution happens in a bubblewrap (`bwrap`) sandbox with resource limits
- Benchmark/correctness runs are isolated in subprocesses
- This keeps untrusted generated code from affecting host system state

## Source Project

- Project: CnakeCharmer
- Focus: Cython optimization via tool-using agent traces
- Repository: https://github.com/dleemiller/CnakeCharmer
