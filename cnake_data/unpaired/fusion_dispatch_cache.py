"""Lightweight fusion dispatcher with param/shape keyed kernel cache."""

from __future__ import annotations

import functools

import numpy as np


class Fusion:
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__
        self._cache = {}

    def __repr__(self):
        return f"<Fusion name={self.name}>"

    def clear_cache(self):
        self._cache = {}

    def __call__(self, *args, **kwargs):
        exec_array = any(isinstance(a, np.ndarray) for a in args)
        if not exec_array:
            return self.func(*args, **kwargs)

        params_info = []
        shape_info = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                params_info.append(arg.dtype.char)
                params_info.append(arg.ndim)
                shape_info.append(arg.shape)
            elif isinstance(arg, np.generic):
                params_info.append(arg.dtype.char)
            elif arg is None:
                params_info.append(None)
            elif isinstance(arg, float):
                params_info.append("d")
            elif isinstance(arg, int):
                params_info.append("l")
            elif isinstance(arg, bool):
                params_info.append("?")
            elif isinstance(arg, complex):
                params_info.append("D")
            else:
                raise TypeError(f"Unsupported input type {type(arg)}")

        param_key = tuple(params_info)
        shape_key = tuple(shape_info)

        cache_shape, kernel_list = self._cache.get(param_key, (None, None))
        if cache_shape is None:
            cache_shape = {}
            kernel_list = []
            self._cache[param_key] = (cache_shape, kernel_list)

        kernel = cache_shape.get(shape_key)
        if kernel is None:
            # In this pure-Python version, the function itself is the kernel.
            kernel = self.func
            cache_shape[shape_key] = kernel
            kernel_list.append(kernel)

        return kernel(*args, **kwargs)


def fuse(*args, **kwargs):
    def wrapper(f, kernel_name=None):
        return Fusion(f, kernel_name)

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return functools.update_wrapper(wrapper(args[0]), args[0])
    return lambda f: functools.update_wrapper(wrapper(f, *args, **kwargs), f)
