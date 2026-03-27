"""Count character class transitions in generated text.

Keywords: character classification, state machine, transitions, string processing, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def classify_transitions(n: int) -> int:
    """Classify characters and count transitions between character classes.

    Generates a deterministic string of length n, classifies each character
    into one of six classes (UPPER, LOWER, DIGIT, SPACE, PUNCT, OTHER),
    and counts the number of times the class changes between consecutive chars.

    Args:
        n: Length of generated string.

    Returns:
        Number of class transitions.
    """
    UPPER = 0
    LOWER = 1
    DIGIT = 2
    SPACE = 3
    PUNCT = 4
    OTHER = 5

    transitions = 0
    prev_class = -1

    for i in range(n):
        # Deterministic char generation using LCG
        h = ((i * 6364136223846793005 + 1442695040888963407) >> 16) & 0x7F
        if 65 <= h <= 90:
            cur_class = UPPER
        elif 97 <= h <= 122:
            cur_class = LOWER
        elif 48 <= h <= 57:
            cur_class = DIGIT
        elif h == 32 or h == 9 or h == 10:
            cur_class = SPACE
        elif h in (33, 44, 46, 59, 58, 63, 45):
            cur_class = PUNCT
        else:
            cur_class = OTHER

        if prev_class >= 0 and cur_class != prev_class:
            transitions += 1
        prev_class = cur_class

    return transitions
