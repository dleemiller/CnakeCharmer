def count_valid_passwords(policy_text):
    """Validate passwords where exactly one indexed position matches letter."""
    valid = 0

    for line in policy_text.splitlines():
        line = line.strip()
        if not line:
            continue

        left, password = line.split(": ")
        limits, letter = left.split(" ")
        lo_s, hi_s = limits.split("-")
        pos_a = int(lo_s)
        pos_b = int(hi_s)

        count = 0
        if password[pos_a - 1] == letter:
            count += 1
        if password[pos_b - 1] == letter:
            count += 1
        if count == 1:
            valid += 1

    return valid
