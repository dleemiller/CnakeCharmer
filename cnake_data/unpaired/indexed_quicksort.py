"""In-place quicksort that tracks permutation indices."""

from __future__ import annotations


def fast_sort(num_arr, index, begin_flag, end_flag):
    if begin_flag >= end_flag:
        return num_arr, index

    i_left = begin_flag
    i_right = end_flag
    val_flag = num_arr[begin_flag]
    index_flag = index[begin_flag]

    while i_left < i_right:
        while num_arr[i_right] >= val_flag and i_left < i_right:
            i_right -= 1
        num_arr[i_left] = num_arr[i_right]
        index[i_left] = index[i_right]

        while num_arr[i_left] <= val_flag and i_left < i_right:
            i_left += 1
        num_arr[i_right] = num_arr[i_left]
        index[i_right] = index[i_left]

    num_arr[i_left] = val_flag
    index[i_left] = index_flag

    fast_sort(num_arr, index, begin_flag, i_left - 1)
    fast_sort(num_arr, index, i_right + 1, end_flag)
    return num_arr, index
