# Benchmark Report

| Benchmark | Variant | Python (ms) | Cython (ms) | Speedup |
|-----------|---------|-------------|-------------|----------|
| coin_change | cython | 7.944 | 0.054 | 147.5x |
| count_substrings | cython | 12.537 | 0.092 | 136.8x |
| knapsack | cython | 33.792 | 0.274 | 123.3x |
| run_length_encode | cython | 77.821 | 0.858 | 90.7x |
| variance | cython | 50.758 | 0.599 | 84.7x |
| insertion_sort | cython | 365.650 | 4.387 | 83.4x |
| dot_product | cython | 7.472 | 0.101 | 73.8x |
| histogram | cython | 42.600 | 0.609 | 69.9x |
| trapezoidal_integration | cython | 26.958 | 0.389 | 69.3x |
| edit_distance | cython | 83.578 | 1.291 | 64.8x |
| count_inversions | cython | 40.638 | 0.704 | 57.8x |
| selection_sort | cython | 198.879 | 3.562 | 55.8x |
| approx_pi | cython | 18.357 | 0.395 | 46.5x |
| euclidean_distances | cython | 26.409 | 0.580 | 45.6x |
| longest_increasing_subsequence | cython | 0.601 | 0.014 | 43.4x |
| matrix_multiply | cython | 2.668 | 0.073 | 36.6x |
| binary_search_count | cython | 78.350 | 2.843 | 27.6x |
| collatz_lengths | cython | 26.711 | 0.978 | 27.3x |
| mandelbrot_count | cython | 39.709 | 1.501 | 26.5x |
| trie_search | cython | 3.282 | 0.170 | 19.3x |
| convolution_1d | cython | 20.115 | 1.196 | 16.8x |
| bubble_sort | cython | 210.343 | 18.430 | 11.4x |
| primes | cython | 0.668 | 0.061 | 11.0x |
| sieve_of_eratosthenes | cython | 4.050 | 0.395 | 10.2x |
| radix_sort | cython | 21.928 | 2.544 | 8.6x |
| gcd_lcm | cython | 16.434 | 1.918 | 8.6x |
| merge_sort | cython | 43.227 | 5.249 | 8.2x |
| running_mean | cython | 7.701 | 1.152 | 6.7x |
| ewma | cython | 33.568 | 5.433 | 6.2x |
| moving_window_sum | cython | 43.213 | 7.637 | 5.7x |
| euler_totient_sieve | cython | 15.723 | 3.862 | 4.1x |
| cumulative_sum | cython | 25.679 | 7.561 | 3.4x |
| fib | cython | 0.003 | 0.001 | 3.1x |
| prefix_max | cython | 20.189 | 7.125 | 2.8x |
| fizzbuzz | cython | 0.498 | 0.180 | 2.8x |
| pascal_triangle_row | cython | 76.394 | 50.455 | 1.5x |
