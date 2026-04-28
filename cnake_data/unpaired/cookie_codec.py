"""HTTP cookie parse/format helpers."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import quote, unquote


def datetime_to_cookie_format(value):
    return value.strftime("%a, %d %b %Y %H:%M:%S GMT").encode()


def datetime_from_cookie_format(value: bytes):
    value_str = value.decode()
    try:
        return datetime.strptime(value_str, "%a, %d %b %Y %H:%M:%S GMT")
    except ValueError:
        return datetime.strptime(value_str, "%a, %d-%b-%Y %H:%M:%S GMT")


class Cookie:
    def __init__(
        self,
        name: bytes,
        value: bytes,
        expires=None,
        domain=None,
        path=None,
        http_only=False,
        secure=False,
        max_age=None,
        same_site=None,
    ):
        if not name:
            raise ValueError("A cookie name is required")
        self.name = name
        self.value = value
        self.expires = expires
        self._expiration = None
        self.domain = domain
        self.path = path
        self.http_only = http_only
        self.secure = secure
        self.max_age = max_age
        self.same_site = same_site

    @property
    def expiration(self):
        if not self.expires:
            return None
        if self._expiration is None:
            self._expiration = datetime_from_cookie_format(self.expires)
        return self._expiration

    @expiration.setter
    def expiration(self, value):
        self._expiration = value
        self.expires = datetime_to_cookie_format(value) if value else None


def _split_value(raw_value: bytes, separator: bytes):
    rindex = raw_value.rindex(separator)
    if rindex == -1:
        return b"", raw_value
    return raw_value[:rindex], raw_value[rindex + 1 :]


def parse_cookie(raw_value: bytes):
    eq = b"="
    parts = raw_value.split(b"; ")
    if len(parts) == 1:
        parts = raw_value.split(b";")

    name, value = _split_value(parts[0], eq)
    if b" " in value and value.startswith(b'"'):
        value = value.strip(b'"')

    expires = domain = path = max_age = same_site = None
    http_only = secure = False

    for part in parts:
        if eq in part:
            k, v = _split_value(part, eq)
            lk = k.lower()
            if lk == b"expires":
                expires = v
            elif lk == b"domain":
                domain = v
            elif lk == b"path":
                path = v
            elif lk == b"max-age":
                max_age = v
            elif lk == b"samesite":
                same_site = v
        else:
            lp = part.lower()
            if lp == b"httponly":
                http_only = True
            if lp == b"secure":
                secure = True

    return Cookie(
        unquote(name.decode()).encode(),
        unquote(value.decode()).encode(),
        expires,
        domain,
        path,
        http_only,
        secure,
        max_age,
        same_site,
    )


def write_cookie_for_response(cookie: Cookie):
    parts = [quote(cookie.name).encode() + b"=" + quote(cookie.value).encode()]
    if cookie.expires:
        parts.append(b"Expires=" + cookie.expires)
    if cookie.max_age:
        parts.append(b"Max-Age=" + cookie.max_age)
    if cookie.domain:
        parts.append(b"Domain=" + cookie.domain)
    if cookie.path:
        parts.append(b"Path=" + cookie.path)
    if cookie.http_only:
        parts.append(b"HttpOnly")
    if cookie.secure:
        parts.append(b"Secure")
    if cookie.same_site:
        parts.append(b"SameSite=" + cookie.same_site)
    return b"; ".join(parts)
