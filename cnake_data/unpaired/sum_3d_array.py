def sum_3d_array(arr):
    """Sum all values of a 3D nested list/array-like object."""
    i_max = len(arr)
    j_max = len(arr[0]) if i_max else 0
    k_max = len(arr[0][0]) if i_max and j_max else 0

    total = 0
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                total += arr[i][j][k]
    return total
