"""Tokenize generated text by character type, return counts per type.

Keywords: string processing, tokenizer, enum, character classification, benchmark
"""

from cnake_data.benchmarks import python_benchmark

ALPHA = 0
DIGIT = 1
SPACE = 2
PUNCT = 3
OTHER = 4


@python_benchmark(args=(100000,))
def cpdef_enum_token_type(n: int) -> tuple:
    """Classify n deterministically-generated characters into token types.

    char_code = (i * 73 + 17) % 128.
    ALPHA: a-z, A-Z. DIGIT: 0-9. SPACE: 32. PUNCT: .,;:!?-. OTHER: rest.

    Args:
        n: Number of characters to classify.

    Returns:
        Tuple of (alpha_count, digit_count, space_count, punct_count, other_count).
    """
    alpha_count = 0
    digit_count = 0
    space_count = 0
    punct_count = 0
    other_count = 0

    punct_chars = {ord("."), ord(","), ord(";"), ord(":"), ord("!"), ord("?"), ord("-")}

    for i in range(n):
        code = (i * 73 + 17) % 128
        if (65 <= code <= 90) or (97 <= code <= 122):
            alpha_count += 1
        elif 48 <= code <= 57:
            digit_count += 1
        elif code == 32:
            space_count += 1
        elif code in punct_chars:
            punct_count += 1
        else:
            other_count += 1

    return (alpha_count, digit_count, space_count, punct_count, other_count)
