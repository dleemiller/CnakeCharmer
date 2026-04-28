def dominate_check(attrs_1, attrs_2):
    """Return -1 if attrs_1 dominates attrs_2, 1 if reverse, else 0."""
    check_flag = len(attrs_1)
    flag = 0
    for i, attr in enumerate(attrs_1):
        if attr > attrs_2[i]:
            flag += 1
        elif attr < attrs_2[i]:
            flag -= 1
        else:
            check_flag -= 1

    if flag == check_flag:
        return -1
    if flag == -check_flag:
        return 1
    return 0
