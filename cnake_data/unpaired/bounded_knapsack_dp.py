import numpy as np

NITEMS = 50


def test_data(filename):
    items = []
    with open(filename, encoding="utf-8") as fp:
        w, n = [int(x) for x in fp.readline().strip().split(" ")]
        for _ in range(n):
            idx, weight, profit, min_q, max_q, fine, cap = [
                int(x) for x in fp.readline().strip().split(" ")
            ]
            items.append(
                {
                    "idx": idx,
                    "weight": weight,
                    "profit": profit,
                    "min_q": min_q,
                    "max_q": max_q,
                    "fine": fine,
                    "cap": cap,
                }
            )
    return w, n, items


def max_profit(w_cap, n_items, py_items):
    m = np.zeros((n_items + 1, w_cap + 1), dtype=np.int64)
    count = np.zeros((n_items + 1, w_cap + 1), dtype=np.int64)

    for i in range(n_items + 1):
        for w in range(w_cap + 1):
            if i == 0:
                m[i, w] = 0
                count[i, w] = 0
                continue

            cur_item = py_items[i - 1]
            if i == 1:
                count[i, w] = 1

            max_profit_val = m[i - 1, w] - min(
                cur_item["cap"], cur_item["fine"] * (cur_item["min_q"] - 0)
            )
            quantity = min(cur_item["max_q"], w // cur_item["weight"])

            for k in range(quantity + 1):
                profit = k * cur_item["profit"] + m[i - 1, w - k * cur_item["weight"]]
                if k < cur_item["min_q"]:
                    fine = min(cur_item["cap"], cur_item["fine"] * (cur_item["min_q"] - k))
                    profit -= fine
                if profit > max_profit_val:
                    max_profit_val = profit

            m[i, w] = max_profit_val
            for k in range(quantity + 1):
                profit = k * cur_item["profit"] + m[i - 1, w - k * cur_item["weight"]]
                if k < cur_item["min_q"]:
                    fine = min(cur_item["cap"], cur_item["fine"] * (cur_item["min_q"] - k))
                    profit -= fine
                if profit == max_profit_val:
                    count[i, w] += count[i - 1, w - k * cur_item["weight"]]

    return m, count


def run(filename):
    w, n, items = test_data(filename)
    m, count = max_profit(w, n, items)
    return int(m[-1][-1]), int(count[-1][-1])
