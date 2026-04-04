# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count character class transitions using cdef enum (Cython).

Keywords: character classification, cdef enum, state machine, transitions, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef enum CharClass:
    UPPER = 0
    LOWER = 1
    DIGIT = 2
    SPACE = 3
    PUNCT = 4
    OTHER = 5


cdef inline CharClass classify_char(int h) noexcept nogil:
    """Classify a character code into a CharClass enum value."""
    if 65 <= h <= 90:
        return UPPER
    elif 97 <= h <= 122:
        return LOWER
    elif 48 <= h <= 57:
        return DIGIT
    elif h == 32 or h == 9 or h == 10:
        return SPACE
    elif h == 33 or h == 44 or h == 46 or h == 59 or h == 58 or h == 63 or h == 45:
        return PUNCT
    else:
        return OTHER


@cython_benchmark(syntax="cy", args=(100000,))
def classify_transitions(int n):
    """Classify chars using cdef enum and count class transitions."""
    cdef int transitions = 0
    cdef int prev_class = -1
    cdef int cur_class
    cdef int i
    cdef unsigned long long h

    with nogil:
        for i in range(n):
            h = (((<unsigned long long>i * <unsigned long long>6364136223846793005ULL + <unsigned long long>1442695040888963407ULL) >> 16) & 0x7F)
            cur_class = <int>classify_char(<int>h)

            if prev_class >= 0 and cur_class != prev_class:
                transitions += 1
            prev_class = cur_class

    return transitions
