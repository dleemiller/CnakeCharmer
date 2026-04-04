# Benchmark Report

| Category | Benchmark | Variant | Python (ms) | Cython (ms) | Speedup |
|----------|-----------|---------|-------------|-------------|----------|
| cryptography | sha256_blocks | cython | 68.870 | 0.149 | 461.0x |
| numerical | jacobian_inverse | cython | 4.433 | 0.018 | 241.6x |
| image_processing | delta_decode | cython | 129.510 | 0.645 | 200.7x |
| image_processing | central_moments | cython | 12.575 | 0.087 | 144.5x |
| simulation | random_walk_2d | cython | 455.093 | 3.375 | 134.8x |
| string_processing | diagonal_scan | cython | 141.732 | 1.366 | 103.8x |
| string_processing | kmer_frequency | cython | 34.835 | 0.346 | 100.6x |
| numerical | prefix_sum_2d | cython | 162.690 | 1.890 | 86.1x |
| compression | lz4_block | cython | 18.220 | 0.212 | 85.8x |
| dsp | fft_radix2 | cython | 38.818 | 0.476 | 81.6x |
| numerical | sparse_matvec | cython | 87.345 | 1.104 | 79.1x |
| string_processing | sequence_identity | cython | 78.894 | 1.209 | 65.2x |
| numerical | matrix_product | cython | 167.463 | 2.607 | 64.2x |
| numerical | running_stats | cython | 648.059 | 10.388 | 62.4x |
| compression | rle_encode | cython | 23.109 | 0.372 | 62.2x |
| graph | hopcroft_karp | cython | 1.114 | 0.022 | 49.9x |
| graph | bfs_shortest_paths | cython | 25.010 | 0.531 | 47.1x |
| geometry | polygon_winding | cython | 978.697 | 21.390 | 45.8x |
| physics | coulomb_lattice | cython | 69.626 | 1.686 | 41.3x |
| dynamic_programming | longest_increasing_subsequence | cython | 0.680 | 0.018 | 38.6x |
| dynamic_programming | coin_change_count | cython | 1.221 | 0.038 | 32.2x |
| algorithms | binary_search_count | cython | 107.010 | 3.343 | 32.0x |
| statistics | exponential_moving_average | cython | 3.183 | 0.100 | 31.9x |
| numerical | piecewise_interp | cython | 1.150 | 0.041 | 28.4x |
| algorithms | prime_sum | cython | 51.947 | 2.270 | 22.9x |
| statistics | welford_variance | cython | 44.675 | 2.081 | 21.5x |
| optimization | genetic_algorithm | cython | 331.467 | 18.599 | 17.8x |
| geometry | dateline_bbox | cython | 4.743 | 0.280 | 16.9x |
| sorting | merge_sort | cython | 47.859 | 5.237 | 9.1x |
| algorithms | radix_sort | cython | 21.723 | 2.555 | 8.5x |
| string_processing | rolling_hash_distinct | cython | 17.089 | 2.319 | 7.4x |
| algorithms | counting_sort | cython | 78.172 | 15.306 | 5.1x |
