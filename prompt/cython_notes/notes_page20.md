**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 20: Loop Optimization Techniques**

Loops are often the most computationally intensive portions of a program. Optimize the loops will result an overall optimized program.

*   **Overview:** Loop optimization involves minimizing the number of calculations within the loop and maximizing the efficiency of the loop structure itself.

**Loop Optimization Techniques:**

*   **Move Invariant Calculations Outside Loops:** Code run outside a loop runs once, hence results a much cleaner program.
    *   If a value is calculated that does is dependent on the loop, remove such calculations from the loop body to calculate only a constant value once.
    *   *Example:* Instead of calculating `len(data)` inside the loop, compute it once *before* the loop and store it in a variable such as `length_value`.

*   **Declare Loop Indices as `cdef int i`:**
    *   Forces C behavior for better and reliable speeds.
    *   Declaring C types increases loop performance by working directly in C level.

*   **Pre-compute and Store Repeated Expressions:**
    *   If an expression is evaluated with same values again and again, compute it once save it in a variable.
    *   For complex objects also, avoid repeating long lookups. Store lookup results in temporary typed variables for reuse.

*   **Unroll Small Loops with Known Iteration Counts:**
    *   Reduces overhead from looping by performing each computation in source code in a list based structure.
    *   If the number of iterations is small and fixed, manually "unroll" the loop.
    *   *Example:* Instead of `for i in range(3): result += data[i]`, write `result += data[0]; result += data[1]; result += data[2]`.

**General Code Sample:**

```cython
cdef int length_value = len(data)  #Move
cdef int i
result = 0.0

for i in range(length_value): #Declare C Type
    temp = expensive_function(constant_in_loop)*5.0   #Pre-compute

```

**Profiling**

*   **Profiling will show where the program is too slow**
*   **Adding type definitions based on profiling helps the compiler find and optimize the areas that need it**

**Tradeoffs**

*   As with all optimizations, test and measure performance implications. Over optimisation can obfuscate code.
