# Benchmark Report

| Benchmark | Variant | Python Avg (s) | Python Std (s) | Cython Avg (s) | Cython Std (s) | Speedup |
|-----------|-----------|----------------|----------------|----------------|----------------|---------|
| binary_search_count | cython | 0.078239 | 0.001095 | 0.002789 | 0.000059 | 28.05x |
| fib | cython | 0.000003 | 0.000000 | 0.000001 | 0.000000 | 3.18x |
| primes | cython | 0.000674 | 0.000012 | 0.000063 | 0.000001 | 10.63x |
| selection_sort | cython | 0.198539 | 0.001722 | 0.003506 | 0.000068 | 56.64x |
| coin_change | cython | 0.007840 | 0.000130 | 0.000048 | 0.000006 | 164.31x |
| knapsack | cython | 0.032877 | 0.000631 | 0.000283 | 0.000003 | 116.35x |
| longest_increasing_subsequence | cython | 0.000622 | 0.000023 | 0.000015 | 0.000001 | 41.99x |
| fizzbuzz | cython | 0.000489 | 0.000009 | 0.000181 | 0.000002 | 2.70x |
| collatz_lengths | cython | 0.026445 | 0.000490 | 0.000949 | 0.000015 | 27.88x |
| gcd_lcm | cython | 0.016169 | 0.000318 | 0.001931 | 0.000034 | 8.37x |
| pascal_triangle_row | cython | 0.076013 | 0.001011 | 0.049973 | 0.000873 | 1.52x |
| sieve_of_eratosthenes | cython | 0.004105 | 0.000104 | 0.000404 | 0.000010 | 10.17x |
| approx_pi | cython | 0.018888 | 0.000410 | 0.000380 | 0.000005 | 49.65x |
| cumulative_sum | cython | 0.026228 | 0.000544 | 0.009503 | 0.000275 | 2.76x |
| dot_product | cython | 0.007875 | 0.000160 | 0.000101 | 0.000001 | 78.01x |
| ewma | cython | 0.034206 | 0.000990 | 0.007157 | 0.000643 | 4.78x |
| histogram | cython | 0.043571 | 0.000727 | 0.000602 | 0.000012 | 72.36x |
| mandelbrot_count | cython | 0.040960 | 0.004527 | 0.001499 | 0.000031 | 27.33x |
| matrix_multiply | cython | 0.002675 | 0.000062 | 0.000076 | 0.000001 | 35.21x |
| running_mean | cython | 0.007751 | 0.000181 | 0.000994 | 0.000037 | 7.80x |
| trapezoidal_integration | cython | 0.027517 | 0.000632 | 0.000402 | 0.000009 | 68.42x |
| bubble_sort | cython | 0.206935 | 0.002332 | 0.018386 | 0.000336 | 11.26x |
| insertion_sort | cython | 0.365655 | 0.003837 | 0.004346 | 0.000291 | 84.14x |
| merge_sort | cython | 0.043724 | 0.000738 | 0.005155 | 0.000118 | 8.48x |
| count_substrings | cython | 0.012560 | 0.000241 | 0.000094 | 0.000001 | 133.54x |
| edit_distance | cython | 0.084241 | 0.001544 | 0.001305 | 0.000044 | 64.56x |
