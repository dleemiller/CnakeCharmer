# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sort dataclass records by key and compute checksum.

Keywords: algorithms, dataclass, sorting, record, extension type, cython, benchmark
"""

cimport cython
import cython.dataclasses
from cnake_data.benchmarks import cython_benchmark


@cython.dataclasses.dataclass
cdef class Record:
    cdef public int key
    cdef public double value


@cython_benchmark(syntax="cy", args=(50000,))
def dataclass_record_sort(int n):
    """Create n records, sort by key, return checksum."""
    cdef list records = [None] * n
    cdef int i, key
    cdef double value
    cdef unsigned int checksum

    for i in range(n):
        key = (
            (<unsigned int>(<long long>i
             * <long long>2654435761)
             ^ (<unsigned int>i >> 3))
            & <int>0x7FFFFFFF
        )
        value = (
            (<long long>i * <long long>1664525
             + <long long>1013904223) % 100000
        ) / 100.0
        records[i] = Record(key=key, value=value)

    records.sort(key=lambda r: r.key)

    checksum = 0
    cdef Record rec
    for i in range(n):
        rec = <Record>records[i]
        checksum = (
            (checksum * <unsigned int>31
             + <unsigned int>rec.key)
            & <unsigned int>0xFFFFFFFF
        )
    return <long long>checksum
