"""BSON+gzip codec for nested Python/NumPy objects."""

from __future__ import annotations

import gzip

import bson
import numpy as np


def encode_to_bson(np_dict):
    def pack(d):
        if isinstance(d, np.ndarray):
            if d.dtype == np.uint8:
                return bytes(d)
            return {"_x_encType": "numpy", "type": str(d.dtype), "data": d.tolist()}
        if isinstance(d, dict):
            return {k: pack(v) for k, v in d.items()}
        if isinstance(d, tuple):
            return tuple(pack(x) for x in d)
        if isinstance(d, list):
            return [pack(x) for x in d]
        return d

    packed = pack(np_dict)
    if not isinstance(packed, dict):
        packed = {"_x_encType": "dict", "data": packed}
    return gzip.compress(bson.dumps(packed))


def decode_from_bson(enc_bytes: bytes):
    def unpack(d):
        if isinstance(d, dict):
            if "_x_encType" in d:
                if d["_x_encType"] == "numpy":
                    return np.array(d["data"], dtype=d["type"])
                raise ValueError(f"Unknown enctype: {d['_x_encType']}")
            return {k: unpack(v) for k, v in d.items()}
        if isinstance(d, bytes):
            return np.frombuffer(d, dtype=np.uint8)
        if isinstance(d, tuple):
            return tuple(unpack(x) for x in d)
        if isinstance(d, list):
            return [unpack(x) for x in d]
        return d

    loaded = bson.loads(gzip.decompress(enc_bytes))
    if isinstance(loaded, dict) and loaded.get("_x_encType") == "dict":
        loaded = loaded["data"]
    return unpack(loaded)
