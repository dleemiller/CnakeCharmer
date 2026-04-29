"""Geohash encode/decode utilities."""

from __future__ import annotations

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_BITS = [16, 8, 4, 2, 1]


def encode(lat: float, lng: float, precision: int = 12) -> str:
    geohash: list[str] = []
    max_lat, min_lat = 90.0, -90.0
    max_lng, min_lng = 180.0, -180.0
    even = True

    while len(geohash) < precision:
        hash_pos = 0
        for bit in range(5):
            if even:
                mid = (max_lng + min_lng) / 2.0
                if lng > mid:
                    hash_pos |= _BITS[bit]
                    min_lng = mid
                else:
                    max_lng = mid
            else:
                mid = (max_lat + min_lat) / 2.0
                if lat > mid:
                    hash_pos |= _BITS[bit]
                    min_lat = mid
                else:
                    max_lat = mid
            even = not even
        geohash.append(BASE32[hash_pos])

    return "".join(geohash)


def decode(hash_value: str) -> dict:
    max_lat, min_lat = 90.0, -90.0
    max_lng, min_lng = 180.0, -180.0
    even = True

    for ch in hash_value:
        hash_pos = BASE32.index(ch)
        for bit in range(4, -1, -1):
            if even:
                mid = (max_lng + min_lng) / 2.0
                if ((hash_pos >> bit) & 1) == 1:
                    min_lng = mid
                else:
                    max_lng = mid
            else:
                mid = (max_lat + min_lat) / 2.0
                if ((hash_pos >> bit) & 1) == 1:
                    min_lat = mid
                else:
                    max_lat = mid
            even = not even

    lat = (min_lat + max_lat) / 2.0
    lng = (min_lng + max_lng) / 2.0
    return {
        "lat": lat,
        "lng": lng,
        "error": {
            "lat": max_lat - lat,
            "lng": max_lng - lng,
        },
    }
