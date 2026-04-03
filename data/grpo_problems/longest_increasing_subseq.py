def longest_increasing_subseq(n):
    """Find LIS length using patience sorting with binary search (O(n log n)).

    Returns (lis_length, number of binary searches, final tails checksum).
    """
    seq = [0] * n
    for i in range(n):
        seq[i] = (i * 2654435761) & 0xFFFFFFFF

    tails = []
    searches = 0

    for i in range(n):
        val = seq[i]
        lo = 0
        hi = len(tails)
        while lo < hi:
            mid = (lo + hi) >> 1
            if tails[mid] < val:
                lo = mid + 1
            else:
                hi = mid
            searches += 1

        if lo == len(tails):
            tails.append(val)
        else:
            tails[lo] = val

    checksum = 0
    for v in tails:
        checksum = (checksum + v) & 0xFFFFFFFF

    return (len(tails), searches, checksum)
