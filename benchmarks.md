# Benchmark Report

| Category | Benchmark | Variant | Python (ms) | Cython (ms) | Speedup |
|----------|-----------|---------|-------------|-------------|----------|
| cryptography | sha256_blocks | cython | 68.870 | 0.149 | 461.0x |
| image_processing | central_moments | cython | 6.382 | 0.014 | 449.2x |
| numerical | jacobian_inverse | cython | 4.245 | 0.018 | 236.7x |
| image_processing | delta_decode | cython | 131.848 | 0.644 | 204.7x |
| string_processing | kmer_frequency | cython | 1.494 | 0.011 | 134.5x |
| simulation | random_walk_2d | cython | 432.835 | 3.328 | 130.1x |
| string_processing | diagonal_scan | cython | 144.563 | 1.369 | 105.6x |
| numerical | prefix_sum_2d | cython | 163.054 | 1.871 | 87.2x |
| compression | lz4_block | cython | 18.220 | 0.212 | 85.8x |
| dsp | fft_radix2 | cython | 38.818 | 0.476 | 81.6x |
| graph | bfs_shortest_paths | cython | 35.133 | 0.472 | 74.5x |
| numerical | sparse_matvec | cython | 81.855 | 1.164 | 70.3x |
| string_processing | sequence_identity | cython | 75.715 | 1.197 | 63.2x |
| numerical | matrix_product | cython | 159.851 | 2.553 | 62.6x |
| numerical | running_stats | cython | 638.908 | 10.542 | 60.6x |
| compression | rle_encode | cython | 23.336 | 0.398 | 58.6x |
| numerical | piecewise_interp | cython | 67.197 | 1.338 | 50.2x |
| graph | hopcroft_karp | cython | 1.114 | 0.022 | 49.9x |
| geometry | polygon_winding | cython | 1012.351 | 20.894 | 48.5x |
| physics | coulomb_lattice | cython | 68.860 | 1.714 | 40.2x |
| dynamic_programming | longest_increasing_subsequence | cython | 0.680 | 0.018 | 38.6x |
| statistics | exponential_moving_average | cython | 3.406 | 0.099 | 34.2x |
| algorithms | binary_search_count | cython | 107.010 | 3.343 | 32.0x |
| dynamic_programming | coin_change_count | cython | 0.968 | 0.039 | 25.2x |
| statistics | welford_variance | cython | 46.900 | 2.111 | 22.2x |
| algorithms | prime_sum | cython | 40.341 | 1.910 | 21.1x |
| optimization | genetic_algorithm | cython | 331.467 | 18.599 | 17.8x |
| geometry | dateline_bbox | cython | 4.924 | 0.285 | 17.3x |
| sorting | merge_sort | cython | 47.859 | 5.237 | 9.1x |
| algorithms | radix_sort | cython | 21.723 | 2.555 | 8.5x |
| string_processing | rolling_hash_distinct | cython | 16.795 | 2.283 | 7.4x |
| math_problems | sieve_of_eratosthenes | cython | 51.500 | 7.034 | 7.3x |
| algorithms | counting_sort | cython | 78.172 | 15.306 | 5.1x |
