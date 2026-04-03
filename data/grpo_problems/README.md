# GRPO Problems

Plain Python files for GRPO training. No test files, no decorators, no Cython ground truth. The agent receives the Python code and must write Cython + equivalence tests itself.

## Adding new problems

1. Find a Cython file in the Stack v2 DuckDB (`utils/stack_data/stack_cython_1k.duckdb`)
2. Convert it to pure Python (remove cdef, cimport, typed memoryviews, etc.)
3. Save as a `.py` file here with a single public function
4. Add the blob_id to the "Used blob_ids" list below

### Good candidates from Stack v2

```sql
SELECT blob_id, path, length_bytes, content
FROM stack_cython
WHERE length_bytes BETWEEN 200 AND 3000
  AND is_generated = false
  AND extension = 'pyx'
  AND content LIKE '%for %'
  AND content LIKE '%cdef %'
  AND content NOT LIKE '%cdef extern%'
  AND content NOT LIKE '%cdef class%'
  AND content NOT LIKE '%cimport numpy%'
ORDER BY length_bytes ASC
```

### Conversion rules

- Remove `cdef`, `cpdef`, type annotations from function signatures
- Replace `from libc.math cimport sqrt` → `from math import sqrt` (or `import math`)
- Replace `cdef int/double/etc` variable declarations → plain assignment
- Replace `prange` → `range`
- Remove `nogil`, `boundscheck`, `wraparound` directives
- Remove typed memoryviews (`double[:, :]` → plain list/array)
- Function must be self-contained, deterministic, and take simple inputs

## Used blob_ids

These Stack v2 Cython files have already been converted to Python problems. Skip them when adding new ones.

| blob_id | path | output file |
|---------|------|-------------|
| `4b64ebc201e2b766` | profiling/approxe.pyx | approx_e.py |
| `10bd6d8eddb8d828` | Python/Cython/Pi/pi_lib.pyx | compute_pi.py |
| `8f63e124d8c07314` | Slides/.../slow_4.pyx | is_prime.py |
| `047aeaeb99a78cd8` | profile_cython/integrate.pyx | numerical_integrate.py |
| `961a8d6f7bc7d7f5` | cython/project-euler/.../solution.pyx | largest_palindrome_product.py |
| `8ae2d07a3ca3dc5d` | .../hello.pyx (sumOfSquares) | sum_of_powers.py |
| `627da3b1bdc0cdca` | examples/cython_for_ODEs/ode0_cy3.pyx | rk2_ode_solver.py |
| `fdc5cd80c3911bd4` | algorithms/core/cython/algorithms.pyx | fibonacci_sequence.py |

### Not from Stack v2 (written manually)

- connected_components.py
- distance_matrix.py
- gauss_blur_1d.py
- histogram_equalize.py
- levenshtein_distance.py
- longest_increasing_subseq.py
- matrix_chain_order.py
- run_length_encode.py
- sieve_primes.py
- sparse_dot_product.py
- string_matching_count.py
