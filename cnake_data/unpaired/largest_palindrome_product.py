def largest_palindrome_product():
    """Find the largest palindrome made from the product of two 3-digit numbers.

    Checks all products of 3-digit numbers (100-999) and returns the largest
    that is a 6-digit palindrome.
    """
    answer = 0
    for i in range(100, 1000):
        for j in range(i + 1, 1000):
            p = i * j
            if p < 100000:
                continue
            s = str(p)
            if s == s[::-1] and p > answer:
                answer = p
    return answer
