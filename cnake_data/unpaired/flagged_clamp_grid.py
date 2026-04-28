def min_max_clip_2d(
    data,
    flags,
    means,
    stds,
    flag_to_check,
    flag_to_set,
    zscore,
    nodata,
    upper_hard_limit,
    lower_hard_limit,
):
    """Clamp flagged pixels into [mean-z*std, mean+z*std] with hard bounds.

    Mutates ``data`` and ``flags`` in-place and returns them.
    """
    y_shape = len(data)
    x_shape = len(data[0]) if y_shape else 0

    for y in range(y_shape):
        for x in range(x_shape):
            if (flags[y][x] & flag_to_check) != flag_to_check:
                continue

            value = data[y][x]
            if value == nodata:
                continue

            max_allowed = means[y][x] + zscore * stds[y][x]
            min_allowed = means[y][x] - zscore * stds[y][x]

            if max_allowed > upper_hard_limit:
                max_allowed = upper_hard_limit
            if min_allowed < lower_hard_limit:
                min_allowed = lower_hard_limit

            if value > max_allowed:
                data[y][x] = max_allowed
                flags[y][x] |= flag_to_set
            elif value < min_allowed:
                data[y][x] = min_allowed
                flags[y][x] |= flag_to_set

    return data, flags
