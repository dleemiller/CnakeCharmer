# Benchmark Report

| Category | Benchmark | Variant | Python (ms) | Cython (ms) | Speedup |
|----------|-----------|---------|-------------|-------------|----------|
| nn_ops | conv2d | cython | 225.542 | 0.163 | 1383.4x |
| nn_ops | max_pool_1d | cython | 536.906 | 1.702 | 315.5x |
|  | matrix_power | cython | 368.436 | 1.910 | 192.9x |
|  | sobel_edge | cython | 124.167 | 0.693 | 179.2x |
|  | diffusion_2d | cython | 531.125 | 3.143 | 169.0x |
|  | topological_sort_dfs | cython | 48.991 | 0.293 | 167.2x |
|  | matrix_chain | cython | 542.837 | 3.288 | 165.1x |
| nn_ops | relu | cython | 428.369 | 2.664 | 160.8x |
|  | box_blur | cython | 142.130 | 0.895 | 158.8x |
|  | weighted_percentile | cython | 196.112 | 1.249 | 157.1x |
|  | topological_sort | cython | 43.389 | 0.277 | 156.7x |
| diff_equations | finite_difference_laplacian | cython | 478.932 | 3.146 | 152.2x |
|  | egg_drop | cython | 276.572 | 1.861 | 148.6x |
|  | valid_parentheses | cython | 104.680 | 0.724 | 144.6x |
|  | bilinear_interpolation | cython | 162.470 | 1.135 | 143.2x |
|  | count_substrings | cython | 12.294 | 0.089 | 138.8x |
|  | xor_cipher_checksum | cython | 492.130 | 3.616 | 136.1x |
|  | coin_change | cython | 8.119 | 0.061 | 133.2x |
|  | aes_sbox | cython | 205.385 | 1.606 | 127.9x |
|  | gaussian_blur | cython | 268.077 | 2.220 | 120.7x |
|  | knapsack | cython | 33.810 | 0.286 | 118.4x |
|  | longest_common_subsequence | cython | 463.774 | 3.980 | 116.5x |
|  | moving_median | cython | 449.742 | 3.953 | 113.8x |
|  | edit_distance_dp | cython | 337.533 | 2.980 | 113.3x |
|  | wave_equation | cython | 138.827 | 1.235 | 112.4x |
|  | fenwick_tree | cython | 436.039 | 3.895 | 112.0x |
|  | delta_encoding | cython | 482.528 | 4.337 | 111.3x |
|  | hamming_distance_sum | cython | 288.360 | 2.629 | 109.7x |
|  | erosion | cython | 85.898 | 0.784 | 109.6x |
|  | random_walk_distance | cython | 384.753 | 3.613 | 106.5x |
| nn_ops | layer_norm | cython | 163.942 | 1.588 | 103.2x |
|  | z_function | cython | 119.584 | 1.181 | 101.3x |
|  | lz77_compress | cython | 17.785 | 0.180 | 98.6x |
|  | threshold | cython | 13.701 | 0.140 | 98.2x |
|  | heat_diffusion | cython | 39.345 | 0.415 | 94.9x |
|  | longest_common_prefix | cython | 61.063 | 0.644 | 94.8x |
|  | manacher | cython | 173.141 | 1.845 | 93.9x |
|  | run_length_encode | cython | 76.540 | 0.842 | 90.9x |
|  | arithmetic_coding_freq | cython | 109.090 | 1.203 | 90.7x |
| optimization | conjugate_gradient | cython | 15.830 | 0.177 | 89.5x |
|  | variance | cython | 50.054 | 0.580 | 86.3x |
|  | insertion_sort | cython | 366.116 | 4.272 | 85.7x |
|  | palindrome_partition | cython | 152.570 | 1.803 | 84.6x |
|  | wildcard_matching | cython | 614.383 | 7.301 | 84.1x |
|  | linear_regression | cython | 79.996 | 0.957 | 83.6x |
|  | caesar_cipher | cython | 614.097 | 7.362 | 83.4x |
|  | longest_zigzag | cython | 396.252 | 4.823 | 82.2x |
| physics | doppler_shift | cython | 306.177 | 3.860 | 79.3x |
|  | dilation | cython | 108.367 | 1.376 | 78.8x |
|  | subset_sum_count | cython | 34.684 | 0.451 | 76.9x |
|  | dot_product | cython | 7.677 | 0.101 | 75.8x |
|  | histogram | cython | 43.323 | 0.587 | 73.8x |
|  | huffman_frequency | cython | 42.153 | 0.586 | 72.0x |
|  | game_of_life | cython | 444.107 | 6.197 | 71.7x |
| physics | lennard_jones | cython | 372.000 | 5.222 | 71.2x |
|  | floyd_warshall | cython | 88.631 | 1.262 | 70.3x |
|  | trapezoidal_integration | cython | 27.472 | 0.393 | 69.9x |
|  | strongly_connected | cython | 110.653 | 1.590 | 69.6x |
|  | histogram_equalize | cython | 33.286 | 0.480 | 69.3x |
| physics | ideal_gas | cython | 362.126 | 5.231 | 69.2x |
|  | counting_sort_chars | cython | 40.449 | 0.586 | 69.0x |
|  | entropy | cython | 404.977 | 6.087 | 66.5x |
|  | median_filter | cython | 93.416 | 1.408 | 66.3x |
|  | kadane_2d | cython | 250.131 | 3.793 | 65.9x |
|  | chi_squared | cython | 104.405 | 1.587 | 65.8x |
|  | edit_distance | cython | 83.579 | 1.276 | 65.5x |
|  | forest_fire | cython | 174.376 | 2.774 | 62.8x |
|  | pagerank | cython | 118.322 | 1.967 | 60.2x |
|  | count_inversions | cython | 40.424 | 0.690 | 58.6x |
|  | rle_compress_int | cython | 524.082 | 8.943 | 58.6x |
|  | kmp_search | cython | 56.951 | 0.984 | 57.9x |
|  | digit_sum | cython | 345.946 | 6.015 | 57.5x |
|  | max_subarray | cython | 269.523 | 4.738 | 56.9x |
|  | fluid_1d | cython | 167.946 | 3.020 | 55.6x |
|  | selection_sort | cython | 199.720 | 3.625 | 55.1x |
|  | point_in_polygon | cython | 360.835 | 6.587 | 54.8x |
|  | reaction_diffusion | cython | 142.124 | 2.607 | 54.5x |
|  | hash_table_ops | cython | 69.011 | 1.291 | 53.5x |
|  | nbody_energy | cython | 221.807 | 4.157 | 53.4x |
|  | run_length_encoding | cython | 487.266 | 9.244 | 52.7x |
|  | reservoir_sampling | cython | 504.303 | 9.713 | 51.9x |
| dsp | autocorrelation | cython | 252.710 | 4.892 | 51.7x |
| dsp | fir_filter | cython | 486.902 | 9.610 | 50.7x |
|  | bfs_shortest_path | cython | 28.640 | 0.575 | 49.8x |
|  | approx_pi | cython | 19.310 | 0.393 | 49.1x |
|  | partition_equal_sum | cython | 372.208 | 7.639 | 48.7x |
|  | segment_tree | cython | 150.408 | 3.129 | 48.1x |
|  | histogram_2d | cython | 75.296 | 1.604 | 46.9x |
|  | euler_path | cython | 55.051 | 1.174 | 46.9x |
|  | articulation_points | cython | 97.431 | 2.096 | 46.5x |
|  | number_theoretic_transform | cython | 23.147 | 0.502 | 46.1x |
| diff_equations | savitzky_golay | cython | 180.652 | 3.938 | 45.9x |
|  | euclidean_distances | cython | 27.140 | 0.598 | 45.4x |
| nn_ops | softmax_stable | cython | 198.660 | 4.398 | 45.2x |
|  | two_sum_count | cython | 4.584 | 0.104 | 44.2x |
|  | max_subarray_sum | cython | 53.409 | 1.209 | 44.2x |
|  | voronoi_nearest | cython | 467.835 | 10.594 | 44.2x |
|  | exponential_histogram | cython | 157.293 | 3.702 | 42.5x |
|  | closest_pair_distance | cython | 396.509 | 9.335 | 42.5x |
|  | bootstrap_mean | cython | 494.751 | 11.854 | 41.7x |
|  | longest_increasing_subsequence | cython | 0.642 | 0.016 | 41.2x |
|  | longest_palindrome | cython | 145.535 | 3.555 | 40.9x |
| compression | lzw_compress | cython | 52.470 | 1.282 | 40.9x |
|  | ising_model | cython | 201.752 | 4.939 | 40.8x |
| physics | relativistic_energy | cython | 561.093 | 13.849 | 40.5x |
|  | stable_marriage | cython | 368.466 | 9.353 | 39.4x |
|  | cubic_spline_eval | cython | 1.741 | 0.045 | 38.5x |
|  | newton_sqrt | cython | 153.076 | 4.005 | 38.2x |
|  | cycle_detection | cython | 34.330 | 0.908 | 37.8x |
|  | aho_corasick_count | cython | 71.337 | 1.920 | 37.1x |
|  | connected_components | cython | 302.298 | 8.157 | 37.1x |
| optimization | least_squares | cython | 12.179 | 0.344 | 35.4x |
|  | matrix_multiply | cython | 2.597 | 0.074 | 34.9x |
|  | rabin_karp | cython | 141.514 | 4.090 | 34.6x |
|  | max_independent_set | cython | 24.157 | 0.703 | 34.4x |
|  | chinese_remainder | cython | 61.537 | 1.868 | 32.9x |
| optimization | brent_minimize | cython | 416.879 | 12.699 | 32.8x |
|  | bipartite_check | cython | 469.978 | 14.418 | 32.6x |
|  | string_hash_compare | cython | 101.998 | 3.185 | 32.0x |
|  | dijkstra | cython | 434.221 | 13.766 | 31.5x |
|  | cellular_automaton_1d | cython | 365.012 | 11.780 | 31.0x |
|  | tim_sort_merge | cython | 370.532 | 12.315 | 30.1x |
|  | line_segment_intersections | cython | 351.824 | 11.916 | 29.5x |
|  | chebyshev_nodes | cython | 236.574 | 8.029 | 29.5x |
|  | rc4_keystream | cython | 86.938 | 3.008 | 28.9x |
|  | extended_gcd_batch | cython | 329.279 | 11.395 | 28.9x |
|  | binary_search_count | cython | 78.853 | 2.750 | 28.7x |
| physics | orbital_mechanics | cython | 280.175 | 9.880 | 28.4x |
|  | shell_sort | cython | 132.148 | 4.744 | 27.9x |
| optimization | linear_least_squares | cython | 167.329 | 6.037 | 27.7x |
|  | sparse_matrix_vector | cython | 36.719 | 1.334 | 27.5x |
|  | bellman_ford | cython | 40.516 | 1.476 | 27.5x |
|  | word_break | cython | 3.143 | 0.116 | 27.1x |
|  | collatz_lengths | cython | 26.783 | 0.990 | 27.1x |
|  | mandelbrot_count | cython | 40.116 | 1.511 | 26.5x |
|  | polygon_area | cython | 135.142 | 5.180 | 26.1x |
|  | graph_coloring_greedy | cython | 80.591 | 3.162 | 25.5x |
|  | sandpile | cython | 31.486 | 1.272 | 24.8x |
|  | fibonacci_word | cython | 298.392 | 12.206 | 24.4x |
|  | traveling_salesman_dp | cython | 248.008 | 10.844 | 22.9x |
|  | running_variance | cython | 443.288 | 19.685 | 22.5x |
|  | fibonacci_matrix | cython | 517.835 | 24.246 | 21.4x |
| optimization | levenberg_marquardt | cython | 70.822 | 3.378 | 21.0x |
|  | catalan_numbers | cython | 34.444 | 1.670 | 20.6x |
|  | modular_exponentiation | cython | 86.909 | 4.222 | 20.6x |
| diff_equations | crank_nicolson | cython | 135.931 | 6.750 | 20.1x |
|  | climbing_stairs | cython | 520.493 | 26.140 | 19.9x |
|  | trie_search | cython | 3.284 | 0.166 | 19.8x |
|  | introsort | cython | 139.710 | 7.141 | 19.6x |
|  | langtons_ant | cython | 23.293 | 1.194 | 19.5x |
|  | lucas_numbers | cython | 517.070 | 26.940 | 19.2x |
| physics | blackbody_radiation | cython | 81.423 | 4.277 | 19.0x |
|  | quick_sort | cython | 141.703 | 7.472 | 19.0x |
|  | heap_sort | cython | 267.581 | 14.172 | 18.9x |
|  | softmax | cython | 13.015 | 0.698 | 18.7x |
| dsp | downsample | cython | 231.971 | 12.858 | 18.0x |
| nn_ops | sigmoid | cython | 78.173 | 4.334 | 18.0x |
|  | covariance_matrix | cython | 296.027 | 16.593 | 17.8x |
|  | pendulum | cython | 285.783 | 16.696 | 17.1x |
| diff_equations | riemann_sum_left | cython | 446.354 | 28.202 | 15.8x |
|  | tridiagonal_solve | cython | 155.439 | 10.462 | 14.9x |
| diff_equations | adams_bashforth | cython | 337.150 | 24.225 | 13.9x |
| optimization | simplex_nelder_mead | cython | 280.037 | 20.138 | 13.9x |
|  | predator_prey | cython | 26.727 | 1.940 | 13.8x |
|  | kernel_density | cython | 205.606 | 14.934 | 13.8x |
|  | merge_intervals | cython | 327.851 | 25.330 | 12.9x |
| optimization | gradient_descent | cython | 53.375 | 4.156 | 12.8x |
|  | epidemic_sir | cython | 0.042 | 0.003 | 12.5x |
| diff_equations | shooting_method | cython | 198.618 | 15.849 | 12.5x |
|  | great_circle | cython | 36.958 | 2.997 | 12.3x |
| dsp | iir_biquad | cython | 382.060 | 31.770 | 12.0x |
|  | gauss_legendre_pi | cython | 0.003 | 0.000 | 11.9x |
|  | convolution_1d | cython | 20.602 | 1.770 | 11.6x |
|  | polynomial_eval | cython | 396.832 | 34.143 | 11.6x |
| nn_ops | conv1d | cython | 360.409 | 31.086 | 11.6x |
| physics | wave_interference | cython | 43.563 | 3.823 | 11.4x |
|  | bubble_sort | cython | 206.721 | 18.622 | 11.1x |
|  | prime_factorization_sum | cython | 16.478 | 1.515 | 10.9x |
|  | primes | cython | 0.650 | 0.062 | 10.5x |
|  | burrows_wheeler | cython | 7.804 | 0.743 | 10.5x |
|  | numerical_derivative | cython | 157.219 | 15.001 | 10.5x |
|  | runge_kutta | cython | 168.758 | 16.337 | 10.3x |
|  | sieve_of_eratosthenes | cython | 4.213 | 0.412 | 10.2x |
|  | dutch_national_flag | cython | 344.674 | 36.017 | 9.6x |
|  | spearman_correlation | cython | 44.249 | 4.675 | 9.5x |
|  | pearson_correlation | cython | 218.402 | 23.233 | 9.4x |
| dsp | goertzel | cython | 113.758 | 12.687 | 9.0x |
|  | fft_naive | cython | 290.783 | 32.497 | 8.9x |
|  | totient_sum | cython | 266.699 | 29.851 | 8.9x |
| dsp | window_functions | cython | 139.868 | 15.702 | 8.9x |
|  | romberg_integration | cython | 585.987 | 67.525 | 8.7x |
|  | gcd_lcm | cython | 16.322 | 1.941 | 8.4x |
|  | rotating_calipers | cython | 31.897 | 3.815 | 8.4x |
|  | merge_sort | cython | 43.245 | 5.182 | 8.3x |
|  | floyd_cycle | cython | 0.002 | 0.000 | 8.2x |
| diff_equations | riemann_sum_midpoint | cython | 460.064 | 56.261 | 8.2x |
|  | radix_sort | cython | 21.428 | 2.673 | 8.0x |
| diff_equations | euler_method | cython | 371.109 | 47.102 | 7.9x |
| diff_equations | midpoint_method | cython | 338.163 | 43.312 | 7.8x |
|  | convex_hull_area | cython | 46.507 | 6.086 | 7.6x |
|  | sweep_line_closest | cython | 27.272 | 3.785 | 7.2x |
|  | moving_window_sum | cython | 43.673 | 6.856 | 6.4x |
|  | miller_rabin | cython | 69.337 | 10.982 | 6.3x |
|  | running_mean | cython | 7.561 | 1.209 | 6.3x |
|  | minimum_spanning_tree | cython | 57.272 | 9.292 | 6.2x |
|  | pigeonhole_sort | cython | 383.929 | 64.760 | 5.9x |
|  | ewma | cython | 35.596 | 6.464 | 5.5x |
|  | counting_sort | cython | 69.670 | 16.676 | 4.2x |
|  | euler_totient_sieve | cython | 15.812 | 3.808 | 4.2x |
|  | cumulative_sum | cython | 25.780 | 7.695 | 3.4x |
|  | fib | cython | 0.004 | 0.001 | 3.0x |
|  | prefix_max | cython | 19.913 | 6.603 | 3.0x |
|  | patience_sort | cython | 83.838 | 28.796 | 2.9x |
|  | fizzbuzz | cython | 0.519 | 0.182 | 2.9x |
|  | mobius_sieve | cython | 134.520 | 54.126 | 2.5x |
| string_processing | suffix_array_lcp | cython | 160.472 | 70.329 | 2.3x |
|  | pascal_triangle_row | cython | 78.960 | 51.514 | 1.5x |
|  | suffix_array_naive | cython | 7.789 | 5.471 | 1.4x |
|  | max_flow | cython | 0.388 | 0.312 | 1.2x |
| nn_ops | gemm | cython | 5.699 | 11.360 | 0.5x |
