# Cython Feature Coverage Checklist

Tracks which Cython user-guide features are represented in py/cy/test problem triplets.

**Last updated:** 2026-03-26
**Total existing problems:** ~380
**Total proposed new problems:** ~137

---

## A. Extension Types (`cdef class`) — Currently: 6 files

- [x] Basic `cdef class` with typed C attributes (6 files)
- [x] `__cinit__` / `__dealloc__` lifecycle (6 files)
- [x] `cdef` methods on classes (6 files)
- [x] `@property` getters (1 file: particle_bounce)
- [x] `cpdef` methods (1 file: polygon_area_centroid)
- [ ] `__getitem__` / `__setitem__` / `__len__` (sequence protocol) — **5 problems**
- [ ] `__iter__` / `__next__` (iterator protocol) — **3 problems**
- [ ] `__add__` / `__mul__` / `__neg__` etc. (arithmetic operators) — **4 problems**
- [ ] `__richcmp__` or `__eq__`/`__lt__` (comparison) — **2 problems**
- [ ] `__call__` (callable objects) — **2 problems**
- [ ] `__hash__` — **1 problem**
- [ ] `__contains__` — **1 problem**
- [ ] `cdef class` inheritance (single inheritance) — **3 problems**
- [ ] `cdef readonly` attributes — **2 problems**
- [ ] `@cython.final` on class or method — **2 problems**
- [ ] `@cython.freelist(N)` — **2 problems**
- [ ] `not None` typed parameter checking — **2 problems**
- [ ] Forward declaration of extension types — **1 problem**
- [ ] `@cython.dataclasses.dataclass` — **2 problems**

**Subtotal: ~32 new problems**

---

## B. Enums — Currently: 2 files

- [x] Basic `cdef enum` (2 files: classify_transitions, rpn_eval)
- [ ] `cpdef enum` (Python-accessible, PEP 435-style) — **3 problems**
- [ ] Anonymous enum (named constants without type) — **2 problems**

**Subtotal: ~5 new problems**

---

## C. Typed Memoryviews — Currently: 3 files (pointer-cast only)

- [x] 1D memoryview from pointer cast `<double[:n]>ptr` (3 files)
- [ ] 2D memoryview `double[:, :]` (strided) — **4 problems**
- [ ] C-contiguous `double[::1]` / `double[:, ::1]` — **3 problems**
- [ ] Fortran-contiguous `double[::1, :]` — **2 problems**
- [ ] `const` memoryviews (read-only views) — **2 problems**
- [ ] Memoryview slicing (`view[2:10]`, subviews) — **2 problems**
- [ ] `.copy()` / `.copy_fortran()` — **1 problem**
- [ ] `.T` transpose — **1 problem**
- [ ] `cython.view.array` for standalone allocation — **2 problems**
- [ ] Pass to C via `&view[0]` — **2 problems**

**Subtotal: ~19 new problems**

---

## D. Fused Types (Generics) — Currently: 0 files

- [ ] `ctypedef fused` basic (int/float/double dispatch) — **3 problems**
- [ ] Fused types with memoryviews — **2 problems**
- [ ] Type checking branches (`if fused_type is int:`) — **2 problems**
- [ ] Built-in fused types (`cython.integral`, `cython.floating`, `cython.numeric`) — **2 problems**

**Subtotal: ~9 new problems**

---

## E. Parallelism (`prange` / `nogil`) — Currently: 0 `prange`, ~17 `nogil` (qsort comparators only)

- [ ] `prange` basic parallel for loop — **4 problems**
- [ ] `prange` with reductions (`+=`, `*=`) — **3 problems**
- [ ] `with nogil:` blocks (release GIL for C computation) — **4 problems**
- [ ] `prange` with `schedule` (static/dynamic/guided) — **2 problems**

**Subtotal: ~13 new problems** (requires `-fopenmp` compile flag)

---

## F. Structs (advanced) — Currently: 9 files

- [x] Basic `cdef struct` with typed fields (9 files)
- [x] Structs with `qsort` comparators (9 files)
- [ ] Nested structs — **2 problems**
- [ ] Packed structs (`cdef packed struct`) — **1 problem**
- [ ] Struct ↔ dict auto-conversion — **2 problems**
- [ ] Struct as function return — **2 problems**

**Subtotal: ~7 new problems**

---

## G. Unions — Currently: 0 files

- [ ] `cdef union` (tagged union / variant type) — **3 problems**

**Subtotal: ~3 new problems**

---

## H. `ctypedef` — Currently: 0 files

- [ ] Type aliases (`ctypedef unsigned long long uint64`) — **3 problems**
- [ ] Function pointer typedefs — **2 problems**

**Subtotal: ~5 new problems**

---

## I. Function Pointers & Callbacks — Currently: ~9 files (qsort only)

- [x] `qsort` with custom comparator (9 files)
- [ ] Custom callback dispatch (non-qsort) — **3 problems**
- [ ] Function pointer arrays (dispatch tables) — **2 problems**
- [ ] Typed function pointer variables — **2 problems**

**Subtotal: ~7 new problems**

---

## J. Error Return Specifications — Currently: 0 files (all use default)

- [ ] `except -1` (specific sentinel) — **2 problems**
- [ ] `except? -1` (check with PyErr_Occurred) — **2 problems**
- [ ] `except *` (always check) — **1 problem**
- [ ] `noexcept` on cdef functions — **2 problems**

**Subtotal: ~7 new problems**

---

## K. Buffer Protocol — Currently: 0 files

- [ ] `__getbuffer__` / `__releasebuffer__` on cdef class — **3 problems**

**Subtotal: ~3 new problems**

---

## L. `cpdef` Functions (standalone) — Currently: 0 standalone

- [ ] `cpdef` module-level functions — **3 problems**

**Subtotal: ~3 new problems**

---

## M. C-tuples — Currently: 0 files

- [ ] `ctuple` return types `(double, int)` — **2 problems**

**Subtotal: ~2 new problems**

---

## N. C++ Interop — Currently: 0 in problem files

- [ ] STL containers (`libcpp.vector`, `libcpp.map`, `libcpp.set`) — **4 problems**
- [ ] `cdef cppclass` wrapping — **2 problems**
- [ ] C++ exception handling (`except +`) — **2 problems**
- [ ] C++ templates — **2 problems**
- [ ] C++ scoped enums (`enum class`) — **1 problem**

**Subtotal: ~11 new problems** (requires `language=c++` build support)

---

## O. Miscellaneous

- [ ] C arrays (fixed-size stack-allocated `cdef int[256]`) — **3 problems**
- [ ] `realloc` / `calloc` usage — **2 problems**
- [ ] `from libc.string cimport memcpy/memset/memcmp` — **3 problems**
- [ ] `@property` with setter — **2 problems**
- [ ] `cdef extern from "header.h"` custom C integration — **2 problems**

**Subtotal: ~12 new problems**

---

## Summary Table

| Category | Current | Proposed | Priority |
|----------|---------|----------|----------|
| A. Extension type features | 6 | ~32 | Tier 1 |
| B. Enums | 2 | ~5 | Tier 2 |
| C. Typed memoryviews | 3 | ~19 | Tier 1 |
| D. Fused types | 0 | ~9 | Tier 1 |
| E. Parallelism (prange/nogil) | 0 | ~13 | Tier 1 |
| F. Structs (advanced) | 9 | ~7 | Tier 2 |
| G. Unions | 0 | ~3 | Tier 3 |
| H. ctypedef | 0 | ~5 | Tier 2 |
| I. Function pointers | 9 | ~7 | Tier 2 |
| J. Error return specs | 0 | ~7 | Tier 2 |
| K. Buffer protocol | 0 | ~3 | Tier 3 |
| L. cpdef functions | 0 | ~3 | Tier 3 |
| M. C-tuples | 0 | ~2 | Tier 3 |
| N. C++ interop | 0 | ~11 | Tier 3 |
| O. Miscellaneous | varies | ~12 | Tier 3 |
| **Total** | | **~137** | |

## Build Changes Required

- **prange/OpenMP (E):** Add `-fopenmp` to `extra_compile_args` and `extra_link_args`
- **C++ interop (N):** Add `language='c++'` to Extension() or create `cy_cpp/` directory
- **Everything else:** Works with existing build system
