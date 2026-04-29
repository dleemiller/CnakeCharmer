"""Memoization decorator optionally scoped per device id."""

from __future__ import annotations

import atexit
import functools

_memos = []


def memoize(for_each_device=False):
    def decorator(f):
        memo = {}
        _memos.append(memo)

        @functools.wraps(f)
        def ret(*args, **kwargs):
            dev_id = -1
            if for_each_device:
                # fallback-friendly hook: caller may monkeypatch device.get_device_id
                from cupy.cuda import device  # type: ignore

                dev_id = device.get_device_id()
            arg_key = (dev_id, args, frozenset(kwargs.items()))
            if arg_key in memo:
                return memo[arg_key]
            result = f(*args, **kwargs)
            memo[arg_key] = result
            return result

        return ret

    return decorator


@atexit.register
def clear_memo():
    for memo in _memos:
        memo.clear()
