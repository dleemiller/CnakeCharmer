# Cython Feature Coverage Checklist

Tracks which Cython user-guide features are represented in py/cy/test problem triplets.

**Last updated:** 2026-03-26
**Total problems:** 478 matched pairs
**Categories complete:** 13 of 15
**Remaining gaps:** ~24 (prange/nogil + C++ interop)

---

## A. Extension Types (`cdef class`) — **Complete**

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
- [x] `@cython.freelist(N)` (freelist_point_sum, freelist_pair_hash)
- [x] `not None` parameter checking (not_none_matrix_sum, not_none_transform)
- [x] Forward declaration (forward_decl_tree_sum)
- [x] `@cython.dataclasses.dataclass` (dataclass_record_sort, dataclass_point_distance)

---

## B. Enums — **Complete**

- [x] Basic `cdef enum` (classify_transitions, rpn_eval)
- [x] `cpdef enum` (cpdef_enum_direction, cpdef_enum_token_type, cpdef_enum_color_blend)
- [x] Anonymous `cdef enum:` (anon_enum_flags, anon_enum_limits)

---

## C. Typed Memoryviews — **Complete**

- [x] 1D memoryview from pointer cast `<double[:n]>ptr` (memview_weighted_sum, matrix_power_trace, image_flip_checksum)
- [x] 2D memoryview via `cython.view.array` (memview_mat_transpose, memview_gauss_blur_2d, memview_mat_add, memview_game_of_life)
- [x] C-contiguous `double[::1]` (contig_prefix_sum, contig_moving_avg, contig_threshold_count)
- [x] Fortran-contiguous `double[::1, :]` (fortran_mat_col_sum, fortran_mat_scale)
- [x] `const` memoryviews (const_dot_product, const_histogram)
- [x] Memoryview slicing (memview_slice_reverse)
- [x] `.copy()` (memview_copy_transform)
- [x] `.T` transpose (memview_transpose_trace)
- [x] `cython.view.array` allocation (memview_mat_transpose, memview_mat_add, etc.)
- [x] Pass to C via `&view[0]` (memview_pass_to_c)

---

## D. Fused Types (Generics) — **Complete**

- [x] `ctypedef fused` basic (fused_array_sum, fused_minmax, fused_clamp, fused_accumulate)
- [x] Float/double helpers (fused_dot_product)
- [x] Fused types with memoryview parameters (fused_memview_sum, fused_memview_scale)
- [x] Type checking branches (fused_type_check_abs)
- [x] Generic algorithm with fused type (fused_search)

---

## E. Parallelism (`prange` / `nogil`) — Not Started

- [ ] `prange` basic parallel for loop — **4 problems**
- [ ] `prange` with reductions (`+=`, `*=`) — **3 problems**
- [ ] `with nogil:` blocks (release GIL for C computation) — **4 problems**
- [ ] `prange` with `schedule` (static/dynamic/guided) — **2 problems**

**Remaining: ~13 problems** (requires `-fopenmp` compile flag)

---

## F. Structs — **Complete**

- [x] Basic `cdef struct` with typed fields (spearman_rank, convex_hull_area, etc.)
- [x] Structs with `qsort` comparators
- [x] Nested structs (nested_struct_particle, nested_struct_rect)
- [x] Packed structs (packed_struct_pixel)
- [x] Struct ↔ dict auto-conversion (struct_to_dict, struct_dict_roundtrip)
- [x] Struct as function return (struct_return_minmax, struct_return_stats)

---

## G. Unions — **Complete**

- [x] `cdef union` int/float type punning (union_int_float)
- [x] `cdef union` byte array packing (union_color_channels)
- [x] Tagged union with `cdef struct` + `cdef union` (union_tagged_value)

---

## H. `ctypedef` — **Complete**

- [x] Type aliases (typedef_hash_table)
- [x] Fixed-size array typedefs (typedef_matrix_ops)
- [x] Function pointer typedefs (typedef_callback_sort)

---

## I. Function Pointers & Callbacks — **Complete**

- [x] `qsort` with custom comparator (9 existing files)
- [x] Dispatch table with function pointer array (dispatch_table_eval)
- [x] Callback function passing (callback_transform)
- [x] Function pointer for generic algorithm (bisection_root)

---

## J. Error Return Specifications — **Complete**

- [x] `except -1` (except_value_search)
- [x] `except? -1.0` (except_check_sqrt)
- [x] `except *` (except_star_validate)

---

## K. Buffer Protocol — **Complete**

- [x] 1D `__getbuffer__`/`__releasebuffer__` for `double*` (buffer_sum_squares)
- [x] 1D buffer protocol for `unsigned char*` (buffer_byte_histogram)
- [x] 2D buffer protocol with shape/strides (buffer_matrix_trace)

---

## L. `cpdef` Functions (standalone) — **Complete**

- [x] `cpdef long long gcd(...)` (cpdef_gcd_sum)
- [x] `cpdef double clamp(...)` (cpdef_clamp_sum)
- [x] `cpdef bint is_prime(...)` (cpdef_is_prime_count)

---

## M. C-tuples — **Complete**

- [x] `(double, double)` return type (ctuple_minmax)
- [x] `(long long, long long)` return type (ctuple_divmod)

---

## N. C++ Interop — Not Started

- [ ] STL containers (`libcpp.vector`, `libcpp.map`, `libcpp.set`) — **4 problems**
- [ ] `cdef cppclass` wrapping — **2 problems**
- [ ] C++ exception handling (`except +`) — **2 problems**
- [ ] C++ templates — **2 problems**
- [ ] C++ scoped enums (`enum class`) — **1 problem**

**Remaining: ~11 problems** (requires `language=c++` build support)

---

## O. Miscellaneous — **Complete**

- [x] Stack-allocated C arrays (stack_array_sort, stack_lut_transform, stack_matrix_det)
- [x] `realloc` (dynamic_array_grow)
- [x] `calloc` (calloc_histogram)
- [x] `memcpy` (memcpy_block_reverse)
- [x] `memset` (memset_clear_pattern)
- [x] `memcmp` (memcmp_dedup_count)
- [x] `@property` with setter (property_particle_energy, property_bounded_value)
- [x] `cdef extern from "header.h"` (extern_abs_sum, extern_string_ops)

---

## Summary Table

| Category | Status | Remaining |
|----------|--------|-----------|
| A. Extension type features | **Complete** | 0 |
| B. Enums | **Complete** | 0 |
| C. Typed memoryviews | **Complete** | 0 |
| D. Fused types | **Complete** | 0 |
| E. Parallelism (prange/nogil) | Not started | ~13 |
| F. Structs | **Complete** | 0 |
| G. Unions | **Complete** | 0 |
| H. ctypedef | **Complete** | 0 |
| I. Function pointers | **Complete** | 0 |
| J. Error return specs | **Complete** | 0 |
| K. Buffer protocol | **Complete** | 0 |
| L. cpdef functions | **Complete** | 0 |
| M. C-tuples | **Complete** | 0 |
| N. C++ interop | Not started | ~11 |
| O. Miscellaneous | **Complete** | 0 |
| **Total remaining** | | **~24** |

## Build Changes Required

- **prange/OpenMP (E):** Add `-fopenmp` to `extra_compile_args` and `extra_link_args`
- **C++ interop (N):** Add `language='c++'` to Extension() or create `cy_cpp/` directory
