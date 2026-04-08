# Benchmark Report

| Category | Benchmark | Variant | Python (ms) | Cython (ms) | Speedup |
|----------|-----------|---------|-------------|-------------|----------|
| image_processing | sigma_filter_2d | cython | 18.446 | 0.231 | 80.0x |
| numerical | level_set_delta | cython | 6.053 | 0.078 | 78.0x |
| image_processing | homography_warp | cython | 2.009 | 0.029 | 68.9x |
| physics | ray_trace_sequential | cython | 2.233 | 0.033 | 66.8x |
| image_processing | lut_trilinear | cython | 8.698 | 0.177 | 49.0x |
| diff_equations | lotka_volterra_rk4 | cython | 2.183 | 0.051 | 42.7x |
| image_processing | radial_distortion | cython | 5.432 | 0.131 | 41.5x |
| physics | vignetting_model | cython | 4.853 | 0.130 | 37.4x |
| numerical | kalman_filter_1d | cython | 1.468 | 0.046 | 31.6x |
| physics | snell_refraction | cython | 2.984 | 0.100 | 29.8x |
| image_processing | uv_texture_sample | cython | 4.132 | 0.159 | 25.9x |
| numerical | newton_sqrt_sum | cython | 2.437 | 0.094 | 25.8x |
| image_processing | thin_plate_spline | cython | 3.492 | 0.141 | 24.7x |
| numerical | ransac_homography | cython | 6.606 | 0.305 | 21.6x |
| physics | camera_response | cython | 13.677 | 0.637 | 21.5x |
| image_processing | bicubic_interp | cython | 91.464 | 4.425 | 20.7x |
| image_processing | rbf_warp | cython | 3.068 | 0.149 | 20.6x |
| image_processing | mesh_warp | cython | 4.181 | 0.216 | 19.4x |
| numerical | camera_projection | cython | 0.425 | 0.022 | 19.2x |
| image_processing | temporal_iir | cython | 21.630 | 1.137 | 19.0x |
| image_processing | gaussian_psf_aperture | cython | 1.798 | 0.100 | 18.0x |
| optimization | flow_shop_makespan | cython | 0.043 | 0.003 | 16.3x |
| physics | phong_shading | cython | 4.052 | 0.249 | 16.3x |
| numerical | row_l2_normalize | cython | 7.103 | 0.465 | 15.3x |
| physics | lambertian_shading | cython | 3.418 | 0.224 | 15.2x |
| numerical | param_grid_eval | cython | 1.603 | 0.117 | 13.7x |
| numerical | dlt_homography | cython | 0.397 | 0.029 | 13.6x |
| physics | monin_obukhov_stability | cython | 6.374 | 0.469 | 13.6x |
| image_processing | flat_field_correction | cython | 10.163 | 0.798 | 12.7x |
| simulation | logistic_bifurcation | cython | 11.846 | 1.049 | 11.3x |
| numerical | vec3_cross_normalize | cython | 2.584 | 0.310 | 8.3x |
