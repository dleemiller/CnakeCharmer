``Advanced Examples: Python vs. Cython``
===========================================

``Overview``
-----------

Building upon the fundamentals, this page presents advanced examples of Cython implementations juxtaposed with their Python counterparts.  Emphasis is given to features such as custom allocators, advanced C++ integration, implementing custom iterators, and managing complex build processes.

.. note::
    These examples target experienced users. Thorough understanding of both
    Python/C semantics and memory management is expected. All optimizations
    should be considered in conjunction with profiling data.

``Example 1: Custom Memory Allocators``
----------------------------------------

**Aim**: Replace standard memory allocation with a potentially faster or resource-constrained alternative.

**Python**: (Demonstrates the idea. Direct overrides are generally not possible portably.)

.. code-block:: python

    class CustomAllocator:
        def __init__(self, pool_size):
            self.memory_pool = bytearray(pool_size)
            self.offset = 0

        def allocate(self, size):
            if self.offset + size > len(self.memory_pool):
                raise MemoryError("Out of memory in pool")
            ptr = memoryview(self.memory_pool)[self.offset:self.offset + size]
            self.offset += size
            return ptr  # Returns a memoryview

        def deallocate(self, ptr):
            #Simplified: Implement a memory pool reset
            pass # noop
            #In reality we store references to the blocks and check here before allocating memory

**Cython**:

.. code-block:: cython

    cimport cpython.mem
    from cpython.mem cimport PyMem_Malloc, PyMem_Free

    cdef extern void* custom_malloc(size_t size)
    cdef extern void custom_free(void* ptr)

    cdef void* custom_malloc(size_t size):
        # Implementation calls the python allocated memory pool
        return _custom_allocator.allocate(size)

    cdef void custom_free(void* ptr):
      # Implementation calls the python deallocation function
        _custom_allocator.deallocate(ptr)

    cdef _custom_allocator = CustomAllocator(100000000) #100 MB

    def process_data(int a, int b):
        cdef int *data = <int *>custom_malloc(sizeof(int) * 10)
        try:
            for i in range(10):
                data[i] = a + b + i
            # ... use data ...
        finally:
            custom_free(data)

**Performance Comparison**: This example shows how CPython objects may make use of a Cython-implemented memory pool for speed; in essence, replacing the standard library. Can be faster if the allocator is designed properly and reduces overhead compared to the system default, however, it is important to include guards so that you don't run out of allocated memory.

``Example 2: Advanced C++ Integration (Templates and STL)``
-------------------------------------------------------------

**Aim**: Seamlessly use advanced features of C++, such as templated classes and STL containers.

**Python**: (Not applicable. C++ features are generally inaccessible directly.)

**Cython**:

.. code-block:: cython

    from libcpp.vector cimport vector
    from libcpp.algorithm cimport sort

    cdef extern from "<algorithm>" namespace "std":
        void sort[T](vector[T]& container)

    cdef class DataProcessor:
        cdef vector[double] data

        def __init__(self, data_list):
            self.data = vector[double](data_list)

        def sort_data(self):
            sort[double](self.data)  # Calls std::sort
        def get_average(self):
            cdef double sum = 0.0
            for value in self.
                sum += value
            return sum / self.data.size()

**Performance Comparison**:  Cython directly interfaces with C++ STL containers and algorithms, providing C++-like control to numerical and calculation heavy codebases. This avoids costly conversions to Python lists and back again, giving very large performance improvements.

``Example 3: Custom Iterator Implementation``
-------------------------------------------

**Aim**: Implement a highly optimized iterator for a custom data structure.

**Python**:

.. code-block:: python

    class MyIterator:
        def __init__(self, data):
            self.data = data
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.data):
                raise StopIteration
            value = self.data[self.index]
            self.index += 1
            return value

**Cython**:

.. code-block:: cython

    cdef class FastIterator:
        cdef int[:] data
        cdef int index
        cdef int length

        def __init__(self, int[:] data):
            self.data = data
            self.index = 0
            self.length = len(data)

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= self.length:
                raise StopIteration
            value = self.data[self.index]
            self.index += 1
            return value

**Performance Comparison**: By implementing the iterator protocols within a *cdef class* (or equivalent pure python declaration with *@cython.cclass*), one can improve the iteration performance significantly, especially for numerical computations.  The pure-python example pays an allocation cost each time the iterator is called. By pre-defining the types for FastIterator, it can take the place of an iterator implemented in C code with nearly equivalent performance characteristics.

``Example 4: Complex Build Configurations (Multiple Source Files)``
-------------------------------------------------------------

**Aim**: Manage a Cython project with dependencies on multiple C/C++ source files and libraries.
This project layout goes beyond a simple *hello world* and into realistic large-scale projects

**Python**: (Not applicable. Python build configuration is focused largely on Python code.)

**Cython**:

.. code-block:: python
    # setup.py
    from setuptools import setup, Extension
    from Cython.Build import cythonize

    ext_modules = [
        Extension(
            "my_module",
            sources=["my_module.pyx", "cpp_source.cpp", "c_source.c"],  #List ALL sources
            include_dirs=["./include"],
            libraries=["custom_lib"],
            library_dirs=["./lib"],
            extra_compile_args=["-std=c++11"],      #C++ compilation flags
            language="c++",
        )
    ]

    setup(
        name='My Cython Project',
        ext_modules=cythonize(ext_modules),
    )

.. note::
    In this example the C Library consists of source as well as pre-built
    object code. It also showcases use of C++ libraries, so extra parameters
    need to be passed to describe the source and linking.  A well structured 
    build process with dependencies solves the compilation.

**Performance Comparison**:  This is not directly performance related, but proper structuring of a Cython build helps scaling complex mixed-language projects, ensures reproducibility, and facilitates integration with external C/C++ codebases.

``Conclusion``
---------------

These examples provide a glimpse into the expanded capabilities offered by Cython.
Cython empowers developers to achieve remarkable performance optimization through advanced techniques and seamless integration with C/C++, tailored to intricate application scenarios.
