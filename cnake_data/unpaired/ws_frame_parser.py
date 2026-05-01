"""WebSocket frame parser for basic control/data frame extraction."""

from __future__ import annotations


def parse_frames(buf: bytes) -> list[dict]:
    frames: list[dict] = []
    i = 0
    n = len(buf)

    while i + 2 <= n:
        b0 = buf[i]
        b1 = buf[i + 1]
        fin = (b0 >> 7) & 1
        opcode = b0 & 0x0F
        masked = (b1 >> 7) & 1
        plen = b1 & 0x7F
        i += 2

        if plen == 126:
            if i + 2 > n:
                break
            plen = (buf[i] << 8) | buf[i + 1]
            i += 2
        elif plen == 127:
            if i + 8 > n:
                break
            plen = 0
            for _ in range(8):
                plen = (plen << 8) | buf[i]
                i += 1

        mask_key = b""
        if masked:
            if i + 4 > n:
                break
            mask_key = buf[i : i + 4]
            i += 4

        if i + plen > n:
            break

        payload = bytearray(buf[i : i + plen])
        i += plen

        if masked:
            for k in range(len(payload)):
                payload[k] ^= mask_key[k % 4]

        frames.append(
            {
                "fin": fin,
                "opcode": opcode,
                "masked": masked,
                "payload": bytes(payload),
            }
        )

    return frames
