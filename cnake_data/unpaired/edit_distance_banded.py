"""Edit distance with optional banded alignment constraint."""

from __future__ import annotations


def edit_distance(s, t, maxdiff=-1):
    m = len(s)
    n = len(t)
    e = maxdiff

    if e != -1 and abs(m - n) > e:
        return abs(m - n)

    s_bytes = s.encode() if isinstance(s, str) else s
    t_bytes = t.encode() if isinstance(t, str) else t

    si = 0
    while m > 0 and n > 0 and s_bytes[si] == t_bytes[si]:
        si += 1
        m -= 1
        n -= 1

    while m > 0 and n > 0 and s_bytes[si + m - 1] == t_bytes[si + n - 1]:
        m -= 1
        n -= 1

    sv = s_bytes[si : si + m]
    tv = t_bytes[si : si + n]

    costs = list(range(m + 1))
    if e == -1:
        for j in range(1, n + 1):
            prev = costs[0]
            costs[0] += 1
            for i in range(1, m + 1):
                match = 1 if sv[i - 1] == tv[j - 1] else 0
                c = min(prev + 1 - match, costs[i] + 1, costs[i - 1] + 1)
                prev = costs[i]
                costs[i] = c
    else:
        for j in range(1, n + 1):
            stop = min(j + e + 1, m + 1)
            if j <= e:
                prev = costs[0]
                costs[0] += 1
                start = 1
            else:
                start = j - e
                prev = costs[start - 1]

            smallest = maxdiff + 1
            for i in range(start, stop):
                match = 1 if sv[i - 1] == tv[j - 1] else 0
                c = min(prev + 1 - match, costs[i] + 1, costs[i - 1] + 1)
                prev = costs[i]
                costs[i] = c
                if c < smallest:
                    smallest = c
            if smallest > maxdiff:
                return maxdiff + 1

    return costs[m]
