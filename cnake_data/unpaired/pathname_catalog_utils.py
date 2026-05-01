"""Catalog-like pathname container helpers."""

from __future__ import annotations


class CatalogStruct:
    def __init__(self, pathnames=None, record_types=None):
        self._pathnames = [] if pathnames is None else list(pathnames)
        self._record_types = [] if record_types is None else list(record_types)

    def get_pathname_list(self):
        return list(self._pathnames)

    def get_record_type(self):
        return list(self._record_types)

    def number_pathnames(self):
        return len(self._pathnames)


def create_catalog(pathnames, record_types=None):
    return CatalogStruct(pathnames=pathnames, record_types=record_types)
