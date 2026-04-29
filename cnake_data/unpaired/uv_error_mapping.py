"""Map libuv-style error codes to Python exceptions and socket error enums."""

from __future__ import annotations

import os


def strerr(errno):
    return os.strerror(errno)


def convert_python_error(uverr, uv_codes):
    oserr = -uverr
    exc = OSError

    if uverr in (uv_codes.get("UV_EACCES"), uv_codes.get("UV_EPERM")):
        exc = PermissionError
    elif uverr in (uv_codes.get("UV_EAGAIN"), uv_codes.get("UV_EALREADY")):
        exc = BlockingIOError
    elif uverr in (uv_codes.get("UV_EPIPE"), uv_codes.get("UV_ESHUTDOWN")):
        exc = BrokenPipeError
    elif uverr == uv_codes.get("UV_ECONNABORTED"):
        exc = ConnectionAbortedError
    elif uverr == uv_codes.get("UV_ECONNREFUSED"):
        exc = ConnectionRefusedError
    elif uverr == uv_codes.get("UV_ECONNRESET"):
        exc = ConnectionResetError
    elif uverr == uv_codes.get("UV_EEXIST"):
        exc = FileExistsError
    elif uverr == uv_codes.get("UV_ENOENT"):
        exc = FileNotFoundError
    elif uverr == uv_codes.get("UV_EINTR"):
        exc = InterruptedError
    elif uverr == uv_codes.get("UV_EISDIR"):
        exc = IsADirectoryError
    elif uverr == uv_codes.get("UV_ESRCH"):
        exc = ProcessLookupError
    elif uverr == uv_codes.get("UV_ETIMEDOUT"):
        exc = TimeoutError

    return exc(oserr, strerr(oserr))


def convert_socket_error(uverr, uv_to_socket):
    return uv_to_socket.get(uverr, 0)
