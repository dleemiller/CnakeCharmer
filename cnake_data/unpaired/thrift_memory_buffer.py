class TCyMemoryBuffer:
    def __init__(self, value=b"", buf_size=4096):
        self.trans = None
        self._buf = bytearray()
        self._cur = 0
        self._buf_size = int(buf_size)
        if value:
            self.setvalue(value)

    def c_read(self, sz):
        remaining = len(self._buf) - self._cur
        if remaining < sz:
            sz = remaining
        if sz <= 0:
            return b""
        out = bytes(self._buf[self._cur : self._cur + sz])
        self._cur += sz
        return out

    def c_write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._buf.extend(data)
        return len(data)

    def _getvalue(self):
        if self._cur >= len(self._buf):
            return b""
        return bytes(self._buf[self._cur :])

    def _setvalue(self, value):
        if isinstance(value, str):
            value = value.encode("utf-8")
        self.clean()
        self._buf.extend(value)

    def read(self, sz):
        return self.c_read(int(sz))

    def write(self, data):
        return self.c_write(data)

    def is_open(self):
        return True

    def open(self):
        return None

    def close(self):
        return None

    def flush(self):
        return None

    def clean(self):
        self._buf = bytearray()
        self._cur = 0

    def getvalue(self):
        return self._getvalue()

    def setvalue(self, value):
        self._setvalue(value)
