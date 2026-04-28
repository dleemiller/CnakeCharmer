def get_cookie_value(s, key):
    """Extract cookie value from a raw cookie header bytes/string."""
    if s is None:
        return None

    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="ignore")
    if isinstance(key, bytes):
        key = key.decode("utf-8", errors="ignore")

    begin = s.find(key)
    if begin == -1:
        return None

    begin += len(key)
    if begin >= len(s) or s[begin] != "=":
        return None

    begin += 1
    end = s.find(";", begin)
    if end == -1:
        return s[begin:]
    return s[begin:end]


def constant_time_compare(val1, val2):
    """Timing-attack-resistant equality check over same-length byte strings."""
    if len(val1) != len(val2):
        return False

    if isinstance(val1, str):
        val1 = val1.encode()
    if isinstance(val2, str):
        val2 = val2.encode()

    result = 0
    for a, b in zip(val1, val2, strict=False):
        result |= a ^ b
    return result == 0
