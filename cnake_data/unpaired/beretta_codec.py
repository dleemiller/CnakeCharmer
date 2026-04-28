import datetime
import re

import termformat

REGEX_TYPE = type(re.compile("kittens"))
ZERO_HOUR = datetime.datetime(1970, 1, 1, 0, 0)


def encode_datetime(time):
    delta = time - ZERO_HOUR
    seconds = delta.days * 24 * 60 * 60 + delta.seconds
    megaseconds = seconds // 1_000_000
    seconds = seconds % 1_000_000
    microseconds = time.microsecond
    return (":bert", ":time", megaseconds, seconds, microseconds)


def decode_datetime(megaseconds, seconds, microseconds):
    seconds = megaseconds * 1_000_000 + seconds
    timestamp = datetime.datetime.utcfromtimestamp(seconds)
    return timestamp.replace(microsecond=microseconds)


def encode_term(term):
    term_type = type(term)
    if term is True:
        return (":bert", ":true")
    if term is False:
        return (":bert", ":false")
    if term is None:
        return (":bert", ":undefined")
    if term == []:
        return (":bert", ":nil")
    if term_type == tuple:
        return tuple(encode_term(x) for x in term)
    if term_type == list:
        return [encode_term(x) for x in term]
    if term_type == dict:
        return (":bert", ":dict", [encode_term(x) for x in term.items()])
    if term_type == datetime.datetime:
        return encode_datetime(term)
    if term_type == REGEX_TYPE:
        flags = []
        if term.flags & re.VERBOSE:
            flags.append(":extended")
        if term.flags & re.IGNORECASE:
            flags.append(":caseless")
        if term.flags & re.MULTILINE:
            flags.append(":multiline")
        if term.flags & re.DOTALL:
            flags.append(":dotall")
        return (":bert", ":regex", term.pattern, tuple(flags))
    return term


def decode_term(term):
    term_type = type(term)
    if term_type == tuple:
        if term[0] == ":bert":
            value_type = term[1]
            if value_type == ":true":
                return True
            if value_type == ":false":
                return False
            if value_type == ":undefined":
                return None
            if value_type == ":nil":
                return []
            if value_type == ":dict":
                dict_items = term[2]
                if not dict_items:
                    return {}
                items = [[decode_term(key), decode_term(value)] for key, value in dict_items]
                return {key: value for key, value in items}
            if value_type == ":time":
                return decode_datetime(*term[2:])
            if value_type == ":regex":
                flags = 0
                pattern, options = term[2:4]
                if ":extended" in options:
                    flags |= re.VERBOSE
                if ":caseless" in options:
                    flags |= re.IGNORECASE
                if ":multiline" in options:
                    flags |= re.MULTILINE
                if ":dotall" in options:
                    flags |= re.DOTALL
                return re.compile(pattern, flags)
            raise ValueError(f"Invalid BERT type: {value_type}")
        return tuple(decode_term(x) for x in term)
    if term_type == list:
        return [decode_term(x) for x in term]
    return term


def encode(term, compressed=0):
    bert = encode_term(term)
    return termformat.encode(bert, compressed=compressed)


def decode(term):
    bert = termformat.decode(term)
    return decode_term(bert)
