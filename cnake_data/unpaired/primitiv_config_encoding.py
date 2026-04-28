import locale

py_primitiv_utils_encoding = None


def set_encoding(encoding=None):
    global py_primitiv_utils_encoding
    if encoding is None:
        _, encoding = locale.getdefaultlocale()
        if encoding is None:
            encoding = "utf-8"
    py_primitiv_utils_encoding = encoding


def get_encoding():
    return py_primitiv_utils_encoding


def pystr_to_cppstr(s):
    if py_primitiv_utils_encoding is None:
        set_encoding()
    return s.encode(py_primitiv_utils_encoding)


def cppstr_to_pystr(b):
    if py_primitiv_utils_encoding is None:
        set_encoding()
    return b.decode(py_primitiv_utils_encoding)
