"""Per-pixel recursive temporal IIR filter.

Applies an exponential smoothing (first-order IIR) filter across synthetic
video frames, maintaining a per-pixel state buffer. Used in temporal
de-noising and motion-blur reduction pipelines.

Keywords: temporal filter, IIR, recursive filter, exponential smoothing, video processing
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(80, 80, 30, 0.3))
def temporal_iir(rows: int, cols: int, n_frames: int, alpha: float) -> tuple:
    """Apply per-pixel IIR temporal smoothing over n_frames synthetic frames.

    Update rule: state[r][c] = alpha * frame[r][c] + (1-alpha) * state[r][c].
    Frame pixels: sin(r*0.1 + f*0.2) * cos(c*0.1 - f*0.15).

    Args:
        rows: Frame height.
        cols: Frame width.
        n_frames: Number of frames to process.
        alpha: Smoothing coefficient in (0, 1].

    Returns:
        Tuple of (total_final, max_final, min_final) of the final state.
    """
    beta = 1.0 - alpha
    state = [[0.0] * cols for _ in range(rows)]

    for f in range(n_frames):
        for r in range(rows):
            for c in range(cols):
                val = math.sin(r * 0.1 + f * 0.2) * math.cos(c * 0.1 - f * 0.15)
                state[r][c] = alpha * val + beta * state[r][c]

    total_final = 0.0
    max_final = -1e18
    min_final = 1e18
    for r in range(rows):
        for c in range(cols):
            v = state[r][c]
            total_final += v
            if v > max_final:
                max_final = v
            if v < min_final:
                min_final = v

    return (total_final, max_final, min_final)
