# Cython Feature Coverage Checklist

Tracks which Cython user-guide features are represented in py/cy/test problem triplets.

**Last updated:** 2026-03-26
**Total problems:** 454 matched pairs
**Remaining gaps:** ~24 (prange/nogil, C++, a few extension type decorators)

---

## A. Extension Types (`cdef class`) — 30+ files

- [x] Basic `cdef class` with typed C attributes
- [x] `__cinit__` / `__dealloc__` lifecycle
- [x] `cdef` methods on classes
- [x] `@property` getters (particle_bounce)
- [x] `@property` with setter (property_particle_energy, property_bounded_value)
- [x] `cpdef` methods (polygon_area_centroid)
- [x] `__getitem__` / `__setitem__` / `__len__` (sorted_array_search, circular_buffer_sum, bit_array_count, sparse_vector_dot, lookup_table_eval, histogram_bucket)
- [x] `__iter__` / `__next__` (range_iterator_sum, linked_list_sum, fibonacci_iterator)
- [x] `__add__` / `__mul__` / `__neg__` / `__iadd__` (complex_multiply_sum, vector3d_cross_sum, matrix2x2_power, fixed_point_accum)
- [x] `__richcmp__` (priority_queue_sort, interval_overlap_count)
- [x] `__call__` (callable_transform, callable_filter_count)
- [x] `__hash__` (interval_overlap_count)
- [x] `__contains__` (sorted_array_search, bit_array_count)
- [x] `cdef class` inheritance (shape_area_sum, animal_simulation, expression_eval)
- [x] `cdef readonly` attributes (immutable_point_distance, config_lookup)
- [x] `@cython.final` (final_accumulator)
- [ ] `@cython.freelist(N)` — **2 problems**
- [ ] `not None` typed parameter checking — **2 problems**
- [ ] Forward declaration of extension types — **1 problem**
- [ ] `@cython.dataclasses.dataclass` — **2 problems**

**Remaining: ~7 problems**

---

## B. Enums — 5 files

- [x] Basic `cdef enum` (classify_transitions, rpn_eval)
- [x] `cpdef enum` (cpdef_enum_direction, cpdef_enum_token_type, cpdef_enum_color_blend)
- [ ] Anonymous enum (named constants without type) — **2 problems**

**Remaining: ~2 problems**

---

## C. Typed Memoryviews — 15+ files

- [x] 1D memoryview from pointer cast `<double[:n]>ptr` (memview_weighted_sum, matrix_power_trace, image_flip_checksum)
- [x] 2D memoryview `double[:, :]` via `cython.view.array` (memview_mat_transpose, memview_gauss_blur_2d, memview_mat_add, memview_game_of_life)
- [x] C-contiguous `double[::1]` (contig_prefix_sum, contig_moving_avg, contig_threshold_count)
- [x] `const` memoryviews (const_dot_product, const_histogram)
- [x] Memoryview slicing (memview_slice_reverse)
- [x] `.copy()` (memview_copy_transform)
- [x] Pass to C via `&view[0]` (memview_pass_to_c)
- [ ] Fortran-contiguous `double[::1, :]` — **2 problems**
- [ ] `.T` transpose — **1 problem**

**Remaining: ~3 problems**

---

## D. Fused Types (Generics) — 5 files

- [x] `ctypedef fused` basic (fused_array_sum, fused_minmax, fused_clamp, fused_accumulate)
- [x] Separate float/double helpers (fused_dot_product)
- [ ] Fused types with memoryviews — **2 problems**
- [ ] Type checking branches (`if fused_type is int:`) — **2 problems**

**Remaining: ~4 problems**

---

## E. Parallelism (`prange` / `nogil`) — 0 `prange` files

- [ ] `prange` basic parallel for loop — **4 problems**
- [ ] `prange` with reductions (`+=`, `*=`) — **3 problems**
- [ ] `with nogil:` blocks (release GIL for C computation) — **4 problems**
- [ ] `prange` with `schedule` (static/dynamic/guided) — **2 problems**

**Remaining: ~13 problems** (requires `-fopenmp` compile flag)

---

## F. Structs — 9+ files

- [x] Basic `cdef struct` with typed fields (spearman_rank, convex_hull_area, etc.)
- [x] Structs with `qsort` comparators
- [ ] Nested structs — **2 problems**
- [ ] Packed structs (`cdef packed struct`) — **1 problem**
- [ ] Struct ↔ dict auto-conversion — **2 problems**
- [ ] Struct as function return — **2 problems**

**Remaining: ~7 problems**

---

## G. Unions — 3 files

- [x] `cdef union` int/float type punning (union_int_float)
- [x] `cdef union` byte array packing (union_color_channels)
- [x] Tagged union with `cdef struct` + `cdef union` (union_tagged_value)

**Complete**

---

## H. `ctypedef` — 3 files

- [x] Type aliases (typedef_hash_table — `ctypedef unsigned long long uint64`)
- [x] Fixed-size array typedefs (typedef_matrix_ops — `ctypedef double[4] vec4_t`)
- [x] Function pointer typedefs (typedef_callback_sort — `ctypedef int (*compare_fn)(...)`)

**Complete**

---

## I. Function Pointers & Callbacks — 12+ files

- [x] `qsort` with custom comparator (9 existing files)
- [x] Dispatch table with function pointer array (dispatch_table_eval)
- [x] Callback function passing (callback_transform)
- [x] Function pointer for generic algorithm (bisection_root)

**Complete**

---

## J. Error Return Specifications — 2 files

- [x] `except -1` (except_value_search)
- [x] `except? -1.0` (except_check_sqrt)
- [ ] `except *` (always check) — **1 problem**

**Remaining: ~1 problem**

---

## K. Buffer Protocol — 3 files

- [x] 1D `__getbuffer__`/`__releasebuffer__` for `double*` (buffer_sum_squares)
- [x] 1D buffer protocol for `unsigned char*` (buffer_byte_histogram)
- [x] 2D buffer protocol with shape/strides (buffer_matrix_trace)

**Complete**

---

## L. `cpdef` Functions (standalone) — 3 files

- [x] `cpdef long long gcd(...)` (cpdef_gcd_sum)
- [x] `cpdef double clamp(...)` (cpdef_clamp_sum)
- [x] `cpdef bint is_prime(...)` (cpdef_is_prime_count)

**Complete**

---

## M. C-tuples — 2 files

- [x] `(double, double)` return type (ctuple_minmax)
- [x] `(long long, long long)` return type (ctuple_divmod)

**Complete**

---

## N. C++ Interop — 0 in problem files

- [ ] STL containers (`libcpp.vector`, `libcpp.map`, `libcpp.set`) — **4 problems**
- [ ] `cdef cppclass` wrapping — **2 problems**
- [ ] C++ exception handling (`except +`) — **2 problems**
- [ ] C++ templates — **2 problems**
- [ ] C++ scoped enums (`enum class`) — **1 problem**

**Remaining: ~11 problems** (requires `language=c++` build support)

---

## O. Miscellaneous

- [x] Stack-allocated C arrays `cdef int[1024]` (stack_array_sort, stack_lut_transform, stack_matrix_det)
- [x] `realloc` (dynamic_array_grow)
- [x] `calloc` (calloc_histogram)
- [x] `memcpy` (memcpy_block_reverse)
- [x] `memset` (memset_clear_pattern)
- [x] `memcmp` (memcmp_dedup_count)
- [x] `@property` with setter (property_particle_energy, property_bounded_value)
- [x] `cdef extern from "header.h"` (extern_abs_sum, extern_string_ops)

**Complete**

---

## Summary Table

| Category | Problems | Status |
|----------|----------|--------|
| A. Extension type features | ~30 | Nearly complete (7 remaining) |
| B. Enums | 5 | Nearly complete (2 remaining) |
| C. Typed memoryviews | 15+ | Nearly complete (3 remaining) |
| D. Fused types | 5 | Partial (4 remaining) |
| E. Parallelism (prange/nogil) | 0 | Not started (13 remaining) |
| F. Structs (advanced) | 9 | Partial (7 remaining) |
| G. Unions | 3 | **Complete** |
| H. ctypedef | 3 | **Complete** |
| I. Function pointers | 12+ | **Complete** |
| J. Error return specs | 2 | Nearly complete (1 remaining) |
| K. Buffer protocol | 3 | **Complete** |
| L. cpdef functions | 3 | **Complete** |
| M. C-tuples | 2 | **Complete** |
| N. C++ interop | 0 | Not started (11 remaining) |
| O. Miscellaneous | 10 | **Complete** |
| **Total remaining** | | **~48 problems** |

## Build Changes Required

- **prange/OpenMP (E):** Add `-fopenmp` to `extra_compile_args` and `extra_link_args`
- **C++ interop (N):** Add `language='c++'` to Extension() or create `cy_cpp/` directory
- **Everything else:** Works with existing build system
