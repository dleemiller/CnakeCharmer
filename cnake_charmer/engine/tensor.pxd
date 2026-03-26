# TensorView struct for future engine use.
# Header only — no .pyx needed yet.
# Kernels can optionally accept TensorView* instead of raw pointers.

cdef struct TensorView:
    float *data
    int ndim
    int shape[4]       # max 4D: [batch, channels, height, width]
    int strides[4]     # element strides per dimension
    int numel          # total element count
