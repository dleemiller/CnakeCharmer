"""Run-length encoded getitem/getlocs helpers."""

from __future__ import annotations

import numpy as np


def getitem(runs, values, start, end):
    arr_length = 100
    nfound = 0
    rsum = 0
    started = 0

    values_arr = np.zeros(arr_length, dtype=float)
    runs_arr = np.zeros(arr_length, dtype=np.int64)

    for i in range(len(runs)):
        r = runs[i]
        rsum += r

        if started == 0:
            if rsum > start:
                if not rsum > end:
                    l = rsum - start
                    runs_arr[nfound] = l
                    values_arr[nfound] = values[i]
                    nfound += 1
                else:
                    return np.array([end - start]), np.array([values[i]])
                started = 1
        else:
            if nfound >= arr_length:
                arr_length *= 2
                values_arr = np.resize(values_arr, arr_length)
                runs_arr = np.resize(runs_arr, arr_length)

            if rsum < end:
                l = runs[i]
                runs_arr[nfound] = l
                values_arr[nfound] = values[i]
                nfound += 1
            else:
                l = runs[i] - (rsum - end)
                runs_arr[nfound] = l
                values_arr[nfound] = values[i]
                nfound += 1
                break

    return runs_arr[:nfound], values_arr[:nfound]


def getlocs(runs, values, locs):
    out = np.zeros(len(locs), dtype=float)
    i = j = 0
    cumsum = 0
    while i < len(runs) and j < len(locs):
        cumsum += runs[i]
        while j < len(locs) and locs[j] < cumsum:
            out[j] = values[i]
            j += 1
        i += 1
    return out
