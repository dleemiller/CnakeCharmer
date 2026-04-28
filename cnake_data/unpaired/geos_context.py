import warnings

from shapely import GEOSException


class get_geos_handle:
    """Lightweight Python context wrapper mirroring GEOS handle lifecycle."""

    def __enter__(self):
        self.handle = object()
        self.last_error = ""
        self.last_warning = ""
        return self.handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.last_error:
            raise GEOSException(self.last_error)
        if self.last_warning:
            warnings.warn(self.last_warning)
        return False
