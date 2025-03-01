# Advanced Examples: Python vs. Cython

## Overview

This document presents advanced Cython implementations compared to their Python counterparts. We'll explore features including custom memory allocators, C++ integration, custom iterators, and complex build configurations.

> **Note**: These examples target experienced users with thorough understanding of Python/C semantics and memory management. Always base optimizations on profiling data.

## Example 1: Custom Memory Allocators

**Aim**: Replace standard memory allocation with a potentially faster or resource-constrained alternative.

### Python Implementation
```python
class CustomAllocator:
    def __init__(self, pool_size):
        self.memory_pool = bytearray(pool_size)
        self.offset = 0
        self.allocated_blocks = {}  # Track allocated blocks
    
    def allocate(self, size):
        if self.offset + size > len(self.memory_pool):
            raise MemoryError("Out of memory in pool")
        
        start = self.offset
        self.offset += size
        view = memoryview(self.memory_pool)[start:self.offset]
        
        # Store reference to track allocated memory
        self.allocated_blocks[id(view)] = (start, size)
        return view
    
    def deallocate(self, ptr):
        # In a real implementation, you'd have a more sophisticated 
        # memory management strategy (e.g., free lists)
        if id(ptr) in self.allocated_blocks:
            del self.allocated_blocks[id(ptr)]
```

### Cython Implementation
```cython
# custom_allocator.pyx
cimport cython
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# Define a simple memory pool in Cython
cdef class MemoryPool:
    cdef:
        char* pool
        size_t pool_size
        size_t offset
    
    def __cinit__(self, size_t pool_size):
        self.pool = <char*>malloc(pool_size)
        if not self.pool:
            raise MemoryError("Failed to allocate memory pool")
        self.pool_size = pool_size
        self.offset = 0
    
    def __dealloc__(self):
        if self.pool:
            free(self.pool)
    
    cdef void* allocate(self, size_t size) nogil:
        cdef void* ptr
        
        if self.offset + size > self.pool_size:
            return NULL  # Out of memory
        
        ptr = self.pool + self.offset
        self.offset += size
        return ptr
    
    cdef void deallocate(self, void* ptr) nogil:
        # Simple implementation: we don't actually free individual blocks
        # In a real allocator, you'd implement proper free list management
        pass

# Create a global memory pool
cdef MemoryPool _global_pool = MemoryPool(100 * 1024 * 1024)  # 100 MB

# Custom allocator functions that can be used with Cython
cdef void* custom_malloc(size_t size) nogil:
    return _global_pool.allocate(size)

cdef void custom_free(void* ptr) nogil:
    _global_pool.deallocate(ptr)

# Example function using the custom allocator
def process_data(int a, int b):
    cdef int* data = <int*>custom_malloc(sizeof(int) * 10)
    if not 
        raise MemoryError("Failed to allocate memory")
    
    try:
        for i in range(10):
            data[i] = a + b + i
        
        # Calculate sum as example
        cdef int total = 0
        for i in range(10):
            total += data[i]
        return total
    finally:
        custom_free(data)
```

### Performance Benefits

A well-designed custom memory allocator can significantly improve performance in memory-intensive applications:

- **Reduced fragmentation**: Better memory layout leading to improved cache utilization
- **Lower overhead**: Faster allocations and deallocations compared to general-purpose allocators
- **Better cache locality**: Related data stored contiguously in memory
- **Controlled memory usage**: Predictable memory footprint and behavior

The key is ensuring robust error handling and proper resource cleanup, which the example demonstrates using `try`/`finally` blocks.

## Example 2: Advanced C++ Integration (Templates and STL)

**Aim**: Seamlessly use advanced C++ features like templated classes and STL containers.

### Python Implementation
Not directly applicable. C++ features are generally inaccessible from pure Python.

### Cython Implementation
```cython
# cpp_integration.pyx
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.algorithm cimport sort, transform
from libcpp.functional cimport function
from libcpp.memory cimport shared_ptr, make_shared
from cython.operator cimport dereference as deref

# Declare C++ template class
cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        vector() except +
        void push_back(T&) except +
        T& operator[](size_t)
        T& at(size_t) except +
        size_t size()
        void clear()
        
        # Iterator support
        cppclass iterator:
            T& operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        
        iterator begin()
        iterator end()

# Standard algorithms
cdef extern from "<algorithm>" namespace "std":
    # Declare sort with specific template parameters
    void sort[Iter](Iter first, Iter last) nogil
    void sort[Iter, Compare](Iter first, Iter last, Compare comp) nogil
    
    # Declare transform for mapping operations
    Iter transform[InputIter, OutputIter, UnaryOp](
        InputIter first, InputIter last, 
        OutputIter result, UnaryOp op) nogil

# Custom C++ comparator as a lambda
cdef extern from *:
    """
    // C++ code in external block
    #include <functional>
    auto desc_compare = [](double a, double b) { return a > b; };
    """
    cdef function[bint(double, double)] desc_compare

cdef class DataProcessor:
    cdef:
        vector[double] data
        shared_ptr[vector[double]] processed_data
    
    def __cinit__(self):
        self.processed_data = make_shared[vector[double]]()
    
    def __init__(self, data_list):
        # Initialize vector from Python list
        cdef double value
        for value in data_list:
            self.data.push_back(value)
    
    def sort_data(self, bint descending=False):
        if descending:
            # Use custom comparator for descending order
            sort(self.data.begin(), self.data.end(), desc_compare)
        else:
            # Use default comparator for ascending order
            sort(self.data.begin(), self.data.end())
    
    def process_data(self, double factor):
        # Clear any previous processed data
        deref(self.processed_data).clear()
        
        # Resize processed data vector to match input size
        deref(self.processed_data).resize(self.data.size())
        
        # Define a C++ lambda for transformation
        cdef function[double(double)] transform_func = \
            lambda double x: x * factor
        
        # Apply transformation using std::transform
        transform(
            self.data.begin(), self.data.end(),
            deref(self.processed_data).begin(),
            transform_func
        )
    
    def get_data(self):
        # Convert C++ vector to Python list
        return [self.data[i] for i in range(self.data.size())]
    
    def get_processed_data(self):
        # Convert processed C++ vector to Python list
        cdef vector[double]* ptr = self.processed_data.get()
        return [deref(ptr)[i] for i in range(deref(ptr).size())]
    
    def get_average(self):
        if self.data.size() == 0:
            return 0.0
            
        cdef double sum = 0.0
        cdef size_t i
        for i in range(self.data.size()):
            sum += self.data[i]
        return sum / self.data.size()
```

### Performance Benefits

Cython's C++ integration provides significant performance advantages:

1. **Direct memory access**: Avoiding Python/C conversions for every data point
2. **Efficient algorithms**: Using optimized C++ STL implementations
3. **Zero-copy operations**: Processing data in-place without creating Python objects
4. **Template specialization**: Leveraging C++ template machinery for type-specific optimizations

This approach is particularly beneficial for data-intensive applications, where the overhead of Python's dynamic typing and memory management can become a bottleneck. Performance gains are often 10-100x faster for numerical operations compared to equivalent Python code.

## Example 3: Custom Iterator Implementation

**Aim**: Implement a highly optimized iterator for a custom data structure.

### Python Implementation
```python
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
```

### Cython Implementation
```cython
# fast_iterator.pyx
cimport cython
from cython.view cimport array as cvarray

# Use freelist to efficiently recycle iterator instances
@cython.freelist(8)
cdef class FastIterator:
    cdef:
        int[:] data        # Memoryview for efficient access
        Py_ssize_t index   # Using proper Py_ssize_t type for indexing
        Py_ssize_t length
        bint reverse       # Flag for reverse iteration
    
    def __cinit__(self, int[:] data, bint reverse=False):
        self.data = data
        self.length = data.shape[0]
        self.reverse = reverse
        self.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Efficiently check bounds
        if self.reverse:
            if self.index < 0:
                raise StopIteration
            value = self.data[self.index]
            self.index -= 1
        else:
            if self.index >= self.length:
                raise StopIteration
            value = self.data[self.index]
            self.index += 1
            
        return value
        
    # Reset the iterator to start/end position
    cpdef void reset(self) nogil:
        if self.reverse:
            self.index = self.length - 1
        else:
            self.index = 0
            
    # C-level next method for use in nogil contexts
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int cnext(self) except? -1 nogil:
        cdef int value
        
        if self.reverse:
            if self.index < 0:
                # Signal end of iteration with -1
                return -1
            value = self.data[self.index]
            self.index -= 1
        else:
            if self.index >= self.length:
                return -1
            value = self.data[self.index]
            self.index += 1
            
        return value

# Example function using the Python iterator protocol
def sum_data(int[:] data):
    cdef:
        FastIterator iterator = FastIterator(data)
        int total = 0
        int value
    
    for value in iterator:
        total += value
    
    return total

# Example using the C-level next method with GIL released
def sum_data_nogil(int[:] data):
    cdef:
        FastIterator iterator = FastIterator(data)
        int total = 0
        int value
    
    with nogil:
        while True:
            value = iterator.cnext()
            if value == -1:  # End of iteration
                break
            total += value
    
    return total
```

### Performance Benefits

The Cython iterator implementation offers substantial performance advantages:

1. **Type specialization**: By specifying exact types, Cython generates optimized C code
2. **Memory efficiency**: Using `@cython.freelist` reduces allocation overhead for frequently created iterators
3. **Direct memory access**: Memoryviews provide near-C-speed access to array data
4. **GIL release**: The `nogil` variant allows parallelization in multi-threaded code
5. **Bounds check elimination**: Using `@cython.boundscheck(False)` removes safety checks in performance-critical sections

In real-world scenarios, this optimized iterator can be 10-50x faster than its Python counterpart, especially for large arrays or when used in tight loops. The performance gap widens further when using the `nogil` variants in multi-threaded contexts.

## Example 4: Complex Build Configurations (Multiple Source Files)

**Aim**: Manage a Cython project with dependencies on multiple C/C++ source files and libraries.

### Project Structure

A complex Cython application might be organized like this:

```
myproject/
├── setup.py
├── pyproject.toml
├── CMakeLists.txt            # Optional: For C++ build integration
├── mypackage/
│   ├── __init__.py
│   ├── _version.py
│   ├── module1.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── _core.pyx         # Main Cython module
│   │   ├── _core.pxd         # Declarations for external usage
│   │   ├── helpers.pxi       # Included Cython code
│   │   └── _core_defs.pxd    # C-level declarations
│   └── extensions/
│       ├── __init__.py
│       ├── _ext1.pyx
│       └── _ext2.pyx
├── src/
│   ├── core/
│   │   ├── core_impl.cpp     # C++ implementation files
│   │   ├── core_impl.h
│   │   └── utils.cpp
│   └── common/
│       ├── common.cpp
│       └── common.h
├── include/
│   └── myproject/
│       ├── api.h             # Public C/C++ headers
│       └── config.h
└── tests/
    ├── test_core.py
    └── test_extensions.py
```

### Modern `setup.py`

```python
# setup.py
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import os
import sys
import platform
import subprocess
import numpy as np

# Determine platform-specific settings
is_windows = platform.system() == "Windows"
is_mac = platform.system() == "Darwin"

# Define compiler flags based on platform and build type
extra_compile_args = []
extra_link_args = []

if not is_windows:
    extra_compile_args.extend(['-std=c++17', '-O3', '-Wall', '-Wextra'])
    if is_mac:
        extra_compile_args.extend(['-stdlib=libc++'])
        extra_link_args.extend(['-stdlib=libc++'])
    else:  # Linux
        extra_compile_args.extend(['-fopenmp'])
        extra_link_args.extend(['-fopenmp'])
else:  # Windows
    extra_compile_args.extend(['/std:c++17', '/O2', '/EHsc'])
    extra_link_args.extend(['/std:c++17'])

# Enable debug mode if requested
if '--debug' in sys.argv:
    sys.argv.remove('--debug')
    if is_windows:
        extra_compile_args = ['/Zi', '/Od', '/std:c++17']
        extra_link_args = ['/DEBUG']
    else:
        extra_compile_args = ['-g', '-O0', '-std=c++17']

# Find all C/C++ source files
def find_sources(directory):
    sources = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.c', '.cpp', '.cxx')):
                sources.append(os.path.join(root, file))
    return sources

# Core sources
core_sources = find_sources('src/core')
common_sources = find_sources('src/common')

# Define extensions
extensions = [
    Extension(
        'mypackage.core._core',
        sources=['mypackage/core/_core.pyx'] + core_sources + common_sources,
        include_dirs=[
            'include',
            'src/core',
            'src/common',
            np.get_include(),  # Include NumPy headers
        ],
        library_dirs=['lib'] if os.path.exists('lib') else [],
        libraries=['custom_lib'] if os.path.exists('lib/libcustom_lib.so') else [],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
    ),
    Extension(
        'mypackage.extensions._ext1',
        sources=['mypackage/extensions/_ext1.pyx'] + common_sources,
        include_dirs=['include', 'src/common', np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
    ),
    Extension(
        'mypackage.extensions._ext2',
        sources=['mypackage/extensions/_ext2.pyx'] + common_sources,
        include_dirs=['include', 'src/common', np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
    ),
]

# Custom build_ext command to handle external dependencies
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Check if we need to build any external libraries first
        if not is_windows and not os.path.exists('lib/libcustom_lib.so'):
            print("Building custom library...")
            if not os.path.exists('lib'):
                os.makedirs('lib')
            subprocess.check_call(['cmake', '-B', 'build', '-S', '.'])
            subprocess.check_call(['cmake', '--build', 'build'])
            subprocess.check_call(['cmake', '--install', 'build'])
        
        build_ext.build_extensions(self)

# Cython compiler directives
cython_directives = {
    'language_level': '3',     # Use Python 3 syntax
    'boundscheck': False,      # Disable bounds checking for speed
    'wraparound': False,       # Disable negative indexing for speed
    'initializedcheck': False, # Disable memoryview initialized check
    'cdivision': True,         # Disable Python's division behavior
    'profile': False,          # Enable profiling
}

# Setup configuration
setup(
    name='myproject',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives=cython_directives,
        annotate=True,  # Generate HTML annotations
    ),
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=[
        'numpy>=1.16.0',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)
```

### Modern `pyproject.toml`

```toml
[build-system]
requires = [
    "setuptools>=42",
    "wheel",
    "Cython>=3.0.0",
    "numpy>=1.16.0",
]
build-backend = "setuptools.build_meta"

[tool.cython]
language_level = "3"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

### CMake Integration for C++ Components

```cmake
cmake_minimum_required(VERSION 3.14)
project(myproject VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add include directories
include_directories(include src/core src/common)

# Collect source files
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
file(GLOB_RECURSE COMMON_SOURCES "src/common/*.cpp")

# Create the custom library
add_library(custom_lib SHARED ${CORE_SOURCES} ${COMMON_SOURCES})

# Install the library
install(TARGETS custom_lib
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
    ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/lib)

# Install headers
install(DIRECTORY include/ DESTINATION ${CMAKE_SOURCE_DIR}/include)
```

### Build System Benefits

This comprehensive setup offers several advantages for complex Cython projects:

1. **Cross-platform compatibility**: Handles different compilers and platforms automatically
2. **External dependency management**: Integrates with CMake for building native libraries
3. **Optimized compilation**: Applies platform-specific optimizations and compiler flags
4. **Modular organization**: Separates Python, Cython, and C/C++ code into logical components
5. **Development tools integration**: Supports debugging, profiling, and testing workflows

This approach scales well for large projects with numerous Cython modules and external dependencies, ensuring reproducible builds across different environments.

## Conclusion

These examples illustrate the advanced capabilities that Cython offers for performance-critical applications. Through efficient integration with C/C++ codebases, custom memory management, optimized data structures, and sophisticated build systems, Cython empowers developers to:

1. **Maximize performance** by eliminating Python's overhead in critical paths
2. **Integrate seamlessly** with existing C/C++ libraries and algorithms
3. **Retain Python's usability** while gaining C-level control where needed
4. **Scale efficiently** for complex applications with sophisticated build requirements

While these techniques require deeper understanding of both Python and C semantics, they can yield remarkable performance improvements (often 10-100x) for computationally intensive tasks.

Cython 3 introduces several improvements over previous versions, including better Python 3 support, enhanced typing, and more efficient generated code. By following the patterns demonstrated in these examples, you can leverage these advancements to create high-performance extensions that remain maintainable and Pythonic.
