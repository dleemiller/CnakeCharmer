import datetime


def int_to_boolean(value):
    if value is None:
        return None
    return bool(value)


def to_str(value):
    return str(value) if value is not None else None


def to_float(value):
    return float(value) if value is not None else None


def to_bytes(value, type_name):
    try:
        if isinstance(value, bytes):
            return value
        return str(value).encode("ascii")
    except Exception as e:
        raise ValueError(
            f"Couldn't parse {type_name} string {value!r} - value is not a string."
        ) from e


def str_to_datetime(value):
    if value is None:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace(" ", "T"))
    except Exception as e:
        raise ValueError(f"Couldn't parse datetime string: '{value}'") from e


def str_to_date(value):
    if value is None:
        return None
    try:
        return datetime.date.fromisoformat(str(value))
    except Exception as e:
        raise ValueError(f"Couldn't parse date string: '{value}'") from e


def str_to_time(value):
    if value is None:
        return None
    try:
        return datetime.time.fromisoformat(str(value))
    except Exception as e:
        raise ValueError(f"Couldn't parse time string: '{value}'") from e


class DecimalResultProcessor:
    def __init__(self, type_, format_):
        self.type_ = type_
        self.format_ = format_

    def process(self, value):
        if value is None:
            return None
        return self.type_(self.format_ % value)
