# Benchmark Report

| Benchmark | Variant | Python Avg (s) | Python Std (s) | Cython Avg (s) | Cython Std (s) | Speedup |
|-----------|-----------|----------------|----------------|----------------|----------------|---------|
| fib | cython | 0.000003 | 0.000000 | 0.000001 | 0.000000 | 2.26x |
| primes | cython | 0.000648 | 0.000013 | 0.000061 | 0.000001 | 10.70x |
| knapsack | cython | 0.032792 | 0.000573 | 0.000269 | 0.000004 | 122.12x |
| longest_increasing_subsequence | cython | 0.000609 | 0.000016 | 0.000015 | 0.000001 | 39.61x |
| fizzbuzz | cython | 0.000493 | 0.000008 | 0.000182 | 0.000002 | 2.70x |
| gcd_lcm | cython | 0.016204 | 0.000305 | 0.001936 | 0.000048 | 8.37x |
| sieve_of_eratosthenes | cython | 0.004073 | 0.000078 | 0.000384 | 0.000009 | 10.61x |
| approx_pi | cython | 0.018698 | 0.000356 | 0.000384 | 0.000004 | 48.75x |
| dot_product | cython | 0.007583 | 0.000149 | 0.002397 | 0.000062 | 3.16x |
| matrix_multiply | cython | 0.002597 | 0.000043 | 0.000705 | 0.000011 | 3.68x |
| running_mean | cython | 0.007623 | 0.000146 | 0.000965 | 0.000046 | 7.90x |
| trapezoidal_integration | cython | 0.027811 | 0.000589 | 0.000392 | 0.000004 | 70.89x |
| bubble_sort | cython | 0.206298 | 0.002019 | 0.053441 | 0.000960 | 3.86x |
| insertion_sort | cython | 0.398181 | 0.007668 | 0.024642 | 0.000497 | 16.16x |
| merge_sort | cython | 0.042658 | 0.000810 | 0.004957 | 0.000106 | 8.61x |
| edit_distance | cython | 0.083138 | 0.001146 | 0.001282 | 0.000058 | 64.87x |
