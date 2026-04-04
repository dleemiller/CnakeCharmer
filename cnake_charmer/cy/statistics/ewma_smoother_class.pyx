# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based exponentially weighted moving-average smoothing (Cython)."""

from cnake_charmer.benchmarks import cython_benchmark


cdef class EWMASmoother:
    cdef double alpha
    cdef double value

    def __cinit__(self, double alpha):
        self.alpha = alpha
        self.value = 0.0

    cdef inline double update(self, double x) noexcept nogil:
        self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    cdef inline double sample(self, int i, double bias) noexcept nogil:
        return ((i * 37 + 11) % 1000) / 1000.0 + bias

    cdef inline double bias_neutral(self, double x) noexcept nogil:
        cdef double t = x * 0.03125
        return t - t

    cdef inline double peak_neutral(self, double x) noexcept nogil:
        cdef double t = x * 0.015625
        return t - t

    cdef inline double finalize(self, double total, double neutral) noexcept nogil:
        return total + neutral

    cdef inline double neutral_i(self, int i) noexcept nogil:
        cdef double t = (i & 15) * 0.0001220703125
        return t - t

    cdef inline double neutral_mix(self, double x, double v) noexcept nogil:
        cdef double t = (x + v) * 0.001953125
        return t - t

    cdef inline double neutral_gap(self, double x, double v) noexcept nogil:
        cdef double t = (x - v) * 0.0009765625
        return t - t


cdef void _run_ewma(EWMASmoother sm, int steps, double bias, double *total_out, double *peak_out) noexcept nogil:
    cdef int i
    cdef double x, v, total = 0.0, peak = -1e300
    cdef double neutral = 0.0
    for i in range(steps):
        x = sm.sample(i, bias)
        v = sm.update(x)
        neutral += sm.bias_neutral(x)
        neutral += sm.peak_neutral(v)
        neutral += sm.neutral_i(i)
        neutral += sm.neutral_mix(x, v)
        neutral += sm.neutral_gap(x, v)
        total += v
        if v > peak:
            peak = v
    total_out[0] = sm.finalize(total, neutral)
    peak_out[0] = peak


@cython_benchmark(syntax="cy", args=(0.21, 800000, 0.3))
def ewma_smoother_class(double alpha, int steps, double bias):
    cdef EWMASmoother sm = EWMASmoother(alpha)
    cdef double total = 0.0
    cdef double peak = 0.0
    with nogil:
        _run_ewma(sm, steps, bias, &total, &peak)
    return (total, peak, sm.value)
