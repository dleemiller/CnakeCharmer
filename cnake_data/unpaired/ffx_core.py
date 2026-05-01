"""Core arithmetic and round functions for radix-based FFX-like transform."""

from __future__ import annotations

import hashlib
import hmac
import math


class cFFX:
    def __init__(self, secret, radix, rounds=10, digestmod="sha1"):
        if rounds % 2 != 0:
            raise ValueError(f"rounds must be even, got {rounds}.")
        if rounds < 0 or rounds > 254:
            raise ValueError(f"rounds must be in range(2, 255), got {rounds}.")
        if radix > 255 or radix <= 1:
            raise ValueError(f"radix must be in range(2, 256), got {radix}.")

        self.secret = secret if secret is not None else b""
        self.radix = radix
        self.rounds = rounds
        self.digestmod = digestmod
        self.maxlen = int(hashlib.new(digestmod).digest_size * math.log(256, self.radix))

    def add(self, a, b):
        n = len(a)
        res = bytearray(n)
        for i in range(n):
            res[i] = (a[i] + b[i]) % self.radix
        return res

    def sub(self, a, b):
        n = len(a)
        res = bytearray(n)
        for i in range(n):
            res[i] = (a[i] - b[i]) % self.radix
        return res

    def rr(self, i, s, n):
        msg = int(i).to_bytes(1, "big") + bytes(s)
        res = bytearray(n)
        h = hmac.digest(self.secret, msg, "sha1")
        d = int(h.hex(), 16)
        for j in range(n):
            d, r = divmod(d, self.radix)
            res[j] = r
        return res

    def split(self, data):
        n = len(data)
        s = n // 2
        return bytearray(data[:s]), bytearray(data[s:])
