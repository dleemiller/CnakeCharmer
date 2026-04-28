def apply_compound_dictionary(tokens, compound_dictionary):
    """Merge 3-gram then 2-gram token patterns using a replacement dictionary.

    Args:
        tokens: list of ``word/tag`` strings.
        compound_dictionary: mapping from tuple(words...) -> "merged/tag".
    """
    words = [t.split("/", 1)[0] for t in tokens]
    merged = list(tokens)

    grams = {
        3: {k: v for k, v in compound_dictionary.items() if len(k) == 3},
        2: {k: v for k, v in compound_dictionary.items() if len(k) == 2},
    }

    for n in (3, 2):
        to_remove = set()
        gram_dict = grams[n]

        for key, replacement in gram_dict.items():
            start = 0
            while start <= len(words) - n:
                if tuple(words[start : start + n]) == key:
                    new_word, new_tag = replacement.split("/", 1)
                    words[start] = new_word
                    merged[start] = f"{new_word}/{new_tag}"
                    for idx in range(start + 1, start + n):
                        to_remove.add(idx)
                    start += n
                else:
                    start += 1

        if to_remove:
            merged = [t for i, t in enumerate(merged) if i not in to_remove]
            words = [t.split("/", 1)[0] for t in merged]

    return merged
