import math


def split_route(s, q, n, w_cap, d, max_len, costs):
    v = [0] + [math.inf] * n
    p = [0] * (n + 1)

    for i in range(1, n + 1):
        cost = 0.0
        j = i
        p_dup = [0]
        sum_qp = []

        while True:
            p_dup.append(s[j])
            if len(sum_qp) == 0:
                sum_qp.append(q[p_dup[-1]])
            else:
                sum_qp.append(sum_qp[-1] + q[p_dup[-1]])

            qp_max = max(0, max(sum_qp))
            qp_min = min(sum_qp)
            if qp_min > 0:
                qp_min = 0

            if i == j:
                cost = costs[(0, s[j])] + d[s[j]] + costs[(s[j], 0)]
            else:
                cost = (
                    cost
                    - costs[(s[j - 1], 0)]
                    + costs[(s[j - 1], s[j])]
                    + d[s[j]]
                    + costs[(s[j], 0)]
                )

            if (cost <= max_len) and (qp_max - qp_min <= w_cap):
                if v[i - 1] + cost < v[j]:
                    v[j] = v[i - 1] + cost
                    p[j] = i - 1
                j += 1

            if (j > n) or (cost > max_len) or (qp_max - qp_min > w_cap):
                break

    return p


def extract_vrp(n, s, p):
    trip = [[] for _ in range(1, n + 1)]
    t = 0
    j = n

    while True:
        i = p[j]
        for k in range(i + 1, j + 1):
            trip[t].append(s[k])
        j = i
        t += 1
        if i == 0:
            break

    return trip


def convert_tsp_to_vrp(s, q, n, w_cap, costs, d=None, max_len=math.inf):
    if d is None:
        d = [0 for _ in range(n + 1)]
    p = split_route([0] + s, q, n, w_cap, d, max_len, costs)
    return extract_vrp(n, [0] + s, p)
