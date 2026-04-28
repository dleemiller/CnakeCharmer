import re

PINYIN_TONE_MARK = {
    0: "aoeiuvü",
    1: "āōēīūǖǖ",
    2: "áóéíúǘǘ",
    3: "ǎǒěǐǔǚǚ",
    4: "àòèìùǜǜ",
}
VOWELS = ("a", "o", "e", "ui", "iu")


def convert_pinyin(word, convert):
    if convert == "capitalize":
        return word.capitalize()
    if convert == "lower":
        return word.lower()
    if convert == "upper":
        return word.upper()
    return word


def decode_pinyin(s):
    """Decode numeric-tone pinyin (e.g., 'hao3') into marked form."""
    s = s.lower()
    result = []
    t = ""

    for c in s:
        if "a" <= c <= "z":
            t += c
            continue
        if c == ":":
            if t and t[-1] == "u":
                t = t[:-1] + "ü"
            continue
        if "0" <= c <= "5":
            tone = int(c) % 5
            if tone != 0:
                m = re.search(r"[aoeiuvü]+", t)
                if m is None:
                    t += c
                elif len(m.group(0)) == 1:
                    idx = PINYIN_TONE_MARK[0].index(m.group(0))
                    t = t[: m.start(0)] + PINYIN_TONE_MARK[tone][idx] + t[m.end(0) :]
                else:
                    for num, vowels in enumerate(VOWELS):
                        if vowels in t:
                            t = t.replace(vowels[-1], PINYIN_TONE_MARK[tone][num], 1)
                            break
        result.append(t)
        t = ""

    result.append(t)
    return "".join(result)
