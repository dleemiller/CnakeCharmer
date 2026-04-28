def gini_impurity(y):
    if not y:
        return 0.0
    counts = {}
    for v in y:
        counts[v] = counts.get(v, 0) + 1
    n = float(len(y))
    p2 = 0.0
    for c in counts.values():
        p = c / n
        p2 += p * p
    return 1.0 - p2


def binary_gini_impurity(y):
    if not y:
        return 0.0
    a = 0
    b = 0
    for v in y:
        if v == 0:
            a += 1
        else:
            b += 1
    n = float(len(y))
    return 1.0 - ((a / n) ** 2 + (b / n) ** 2)


def information_gain(y_left, y_right, gini_parent, binary=False):
    yl = float(len(y_left))
    yr = float(len(y_right))
    if yl + yr == 0:
        return 0.0
    wl = yl / (yl + yr)
    wr = 1.0 - wl

    impurity_fn = binary_gini_impurity if binary else gini_impurity
    l_imp = wl * impurity_fn(y_left)
    r_imp = wr * impurity_fn(y_right)
    return gini_parent - l_imp - r_imp


def decision_split(x, decision):
    """Split rows by (feature_index, threshold) using >= criterion."""
    col_pos, decision_val = decision
    true_instances = []
    false_instances = []
    for row in x:
        if row[col_pos] >= decision_val:
            true_instances.append(row)
        else:
            false_instances.append(row)
    return true_instances, false_instances
