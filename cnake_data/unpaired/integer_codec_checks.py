from __future__ import annotations


def encode_int2(val: int) -> int:
    if val < -32768 or val > 32767:
        raise OverflowError("value out of int16 range")
    return val


def encode_int4(val: int) -> int:
    if val < -2147483648 or val > 2147483647:
        raise OverflowError("value out of int32 range")
    return val


def encode_uint4(val: int) -> int:
    if val < 0 or val > 4294967295:
        raise OverflowError("value out of uint32 range")
    return val


def encode_int8(val: int) -> int:
    if val < -9223372036854775808 or val > 9223372036854775807:
        raise OverflowError("value out of int64 range")
    return val
