def pos_redshift_space(pos, vel, box_size, hubble, redshift, axis):
    """Project particle positions into redshift space along one axis.

    Args:
        pos: Mutable sequence of [x, y, z]-like float rows.
        vel: Sequence of velocity rows with same shape as ``pos``.
        box_size: Periodic box size.
        hubble: H(z) scaling term.
        redshift: Cosmological redshift value.
        axis: Axis index to update.
    """
    factor = (1.0 + redshift) / hubble

    for i in range(len(pos)):
        shifted = pos[i][axis] + vel[i][axis] * factor
        if shifted > box_size or shifted < 0.0:
            shifted = (shifted + box_size) % box_size
        pos[i][axis] = shifted

    return pos
