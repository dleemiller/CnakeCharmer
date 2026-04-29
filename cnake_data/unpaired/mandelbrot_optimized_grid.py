"""Optimized Mandelbrot grid computation."""

from __future__ import annotations


def calculate_grid(pim, pre, i_max, t, pim_min, pim_max, pre_min, pre_max):
    t2 = t**4
    im_span = pim_max - pim_min
    re_span = pre_max - pre_min
    im_step = im_span / pim
    re_step = re_span / pre

    ms = []
    for i_im in range(pim):
        im = i_im * im_step + pim_min
        row = []
        for i_re in range(pre):
            c_real = i_re * re_step + pre_min
            c_imag = im
            z_real = 0.0
            z_imag = 0.0
            i = 0
            while (z_real * z_real + z_imag * z_imag) <= t2 and i < i_max:
                nz_real = z_real * z_real - z_imag * z_imag + c_real
                nz_imag = 2.0 * z_real * z_imag + c_imag
                z_real, z_imag = nz_real, nz_imag
                i += 1
            row.append(float(i) / float(i_max))
        ms.append(row)
    return ms
