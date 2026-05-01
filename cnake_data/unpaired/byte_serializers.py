"""Primitive integer serializers/deserializers for byte buffers."""

from __future__ import annotations

import struct


def get_int8(binbuffer, offset):
    val = struct.unpack_from("<b", binbuffer, offset[0])[0]
    offset[0] += 1
    return val


def set_int8(buff, obj):
    buff.extend(struct.pack("<b", int(obj)))


def get_uint8(binbuffer, offset):
    val = struct.unpack_from("<B", binbuffer, offset[0])[0]
    offset[0] += 1
    return val


def set_uint8(buff, obj):
    buff.extend(struct.pack("<B", int(obj)))


def get_int16(binbuffer, offset):
    val = struct.unpack_from("<h", binbuffer, offset[0])[0]
    offset[0] += 2
    return val


def set_int16(buff, obj):
    buff.extend(struct.pack("<h", int(obj)))


def get_uint16(binbuffer, offset):
    val = struct.unpack_from("<H", binbuffer, offset[0])[0]
    offset[0] += 2
    return val


def set_uint16(buff, obj):
    buff.extend(struct.pack("<H", int(obj)))


def get_int32(binbuffer, offset):
    val = struct.unpack_from("<i", binbuffer, offset[0])[0]
    offset[0] += 4
    return val


def set_int32(buff, obj):
    buff.extend(struct.pack("<i", int(obj)))


def get_uint32(binbuffer, offset):
    val = struct.unpack_from("<I", binbuffer, offset[0])[0]
    offset[0] += 4
    return val


def set_uint32(buff, obj):
    buff.extend(struct.pack("<I", int(obj)))


def get_int64(binbuffer, offset):
    val = struct.unpack_from("<q", binbuffer, offset[0])[0]
    offset[0] += 8
    return val


def set_int64(buff, obj):
    buff.extend(struct.pack("<q", int(obj)))


def get_uint64(binbuffer, offset):
    val = struct.unpack_from("<Q", binbuffer, offset[0])[0]
    offset[0] += 8
    return val


def set_uint64(buff, obj):
    buff.extend(struct.pack("<Q", int(obj)))
