"""ODPS-like scalar validator helpers."""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

BIGINT_MIN = -(2**63)
BIGINT_MAX = 2**63 - 1
STRING_LEN_MAX = 8 * 1024 * 1024
DECIMAL_INT_LEN_MAX = 36
DECIMAL_SCALE_MAX = 18
TO_SCALE = Decimal(str(10**-DECIMAL_SCALE_MAX))


def validate_bigint(val):
    val = int(val)
    if BIGINT_MIN <= val <= BIGINT_MAX:
        return val
    raise ValueError(f"InvalidData: Bigint({val}) out of range")


def validate_string(val):
    if isinstance(val, bytes):
        raw = val
    else:
        raw = str(val).encode("utf-8")
    if len(raw) <= STRING_LEN_MAX:
        return raw.decode("utf-8")
    raise ValueError("InvalidData: string too long")


def validate_datetime(val):
    if isinstance(val, datetime):
        return val
    if isinstance(val, (bytes, str)):
        s = val.decode("utf-8") if isinstance(val, bytes) else val
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    raise TypeError(f"Invalid data type: expect datetime, got {type(val)}")


def validate_decimal(val):
    if not isinstance(val, Decimal):
        val = Decimal(str(val))
    scaled = val.quantize(TO_SCALE, ROUND_HALF_UP)
    int_len = len(str(scaled)) - DECIMAL_SCALE_MAX - 1
    if int_len > DECIMAL_INT_LEN_MAX:
        raise ValueError(f"decimal value {val} overflow")
    return scaled
