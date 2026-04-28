import string

PW_CHARACTERS = string.printable[:-5]


def apply_edits(word):
    """Generate unique edit-distance-1 style candidates plus case variants."""
    if not word:
        return [word]

    out = []
    out.append(word.capitalize())
    if word[0].isupper():
        out.append(word[0].lower() + word[1:])
    out.append(word.swapcase())

    # Insertions and replacements
    for ch in PW_CHARACTERS:
        for j in range(len(word)):
            out.append(word[:j] + ch + word[j:])
            if ch != word[j]:
                out.append(word[:j] + ch + word[j + 1 :])
        out.append(word + ch)

    # Deletions
    for j in range(len(word)):
        out.append(word[:j] + word[j + 1 :])

    # Stable dedupe
    seen = set()
    uniq = []
    for candidate in out:
        if candidate not in seen:
            seen.add(candidate)
            uniq.append(candidate)
    return uniq
