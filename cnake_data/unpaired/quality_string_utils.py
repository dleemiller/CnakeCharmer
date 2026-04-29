"""Utility conversions between quality strings and integer arrays."""

from __future__ import annotations

import sys
from array import array


def qualitystring_to_array(input_str, offset=33):
    if input_str is None:
        return None
    return array("B", [i - offset for i in input_str])


def array_to_qualitystring(qualities, offset=33):
    if qualities is None:
        return None
    result = array("B", qualities)
    for x in range(len(qualities)):
        result[x] = qualities[x] + offset
    return result.tobytes()


def qualities_to_qualitystring(qualities, offset=33):
    if qualities is None:
        return None
    elif isinstance(qualities, array):
        return array_to_qualitystring(qualities, offset=offset)
    else:
        return "".join([chr(x + offset) for x in qualities])


def from_string_and_size(s, length):
    if sys.version_info[0] < 3:
        return s[:length]
    else:
        return s[:length].decode("ascii") if isinstance(s, (bytes, bytearray)) else s[:length]


def encode_filename(filename):
    if filename is None:
        return None
    elif isinstance(filename, bytes):
        return filename
    elif isinstance(filename, str):
        enc = sys.getfilesystemencoding() or sys.getdefaultencoding() or "ascii"
        return filename.encode(enc)
    else:
        raise TypeError("Argument must be string or unicode.")
