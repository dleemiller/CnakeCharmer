import math


def string_subseq_kernel(s, t, k, lambda_decay):
    """Compute the normalized string subsequence kernel (SSK) between two strings.

    The SSK measures similarity between strings based on shared
    subsequences of length k, with a decay factor lambda that
    penalizes gaps. Returns a value between 0 and 1, where 1 means
    the strings are identical.

    Args:
        s: First input string.
        t: Second input string.
        k: Subsequence length (positive integer).
        lambda_decay: Gap penalty decay factor in (0, 1].

    Returns:
        Float similarity score in [0, 1].
    """
    if s == t:
        return 1.0
    if min(len(s), len(t)) < k:
        return 0.0

    def get_char_index(c, string):
        return [i for i, letter in enumerate(string) if letter == c]

    def get_k_prime_val(s, t, k, ld):
        m_limit = len(s) + 1
        n_limit = len(t) + 1

        # k_prime_val[i][m][n]
        k_prime_val = [[[0.0] * n_limit for _ in range(m_limit)] for _ in range(k)]
        k_doub_prime_val = [[[0.0] * n_limit for _ in range(m_limit)] for _ in range(k)]

        # Initialize base case: k_prime_val[0] = 1 everywhere
        for m in range(m_limit):
            for n in range(n_limit):
                k_prime_val[0][m][n] = 1.0

        for i in range(1, k):
            for m in range(m_limit):
                for n in range(n_limit):
                    if min(m, n) >= i:
                        if s[m - 1] == t[n - 1]:
                            k_doub_prime_val[i][m][n] = ld * (
                                k_doub_prime_val[i][m][n - 1]
                                + ld * k_prime_val[i - 1][m - 1][n - 1]
                            )
                        else:
                            k_doub_prime_val[i][m][n] = ld * k_doub_prime_val[i][m][n - 1]
                    else:
                        k_prime_val[i][m][n] = 0.0

                    k_prime_val[i][m][n] = ld * k_prime_val[i][m - 1][n] + k_doub_prime_val[i][m][n]

        return k_prime_val

    def get_k_val(s, t, k, ld, k_prime_val):
        k_val = 0.0
        for i in range(len(s) + 1):
            if min(len(s[:i]), len(t)) >= k:
                indices = get_char_index(s[i - 1], t)
                k_val += ld**2 * sum(k_prime_val[k - 1][len(s[:i]) - 1][j] for j in indices)
        return k_val

    kpv_st = get_k_prime_val(s, t, k, lambda_decay)
    k_st = get_k_val(s, t, k, lambda_decay, kpv_st)

    kpv_ss = get_k_prime_val(s, s, k, lambda_decay)
    k_ss = get_k_val(s, s, k, lambda_decay, kpv_ss)

    kpv_tt = get_k_prime_val(t, t, k, lambda_decay)
    k_tt = get_k_val(t, t, k, lambda_decay, kpv_tt)

    denom = math.sqrt(k_ss * k_tt) if k_ss * k_tt else 1e-19
    return k_st / denom
