import datetime


class BaseActivity:
    def __init__(self, **kwargs):
        self._created_at = kwargs.pop("created_at", None)

    @property
    def created_at(self):
        if self._created_at is not None:
            return datetime.datetime.utcfromtimestamp(self._created_at / 1000)
        return None
