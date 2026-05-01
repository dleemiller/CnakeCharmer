"""Feature-map dataset helpers inspired by CREG training structures."""

from __future__ import annotations


class Bias:
    def __repr__(self):
        return "***BIAS***"


BIAS = Bias()


class Intercept:
    def __init__(self, level):
        self.level = int(level)

    def __repr__(self):
        return f"y>={self.level}"


def as_str(name):
    if isinstance(name, bytes):
        return name.decode("utf8")
    if isinstance(name, str):
        return name
    raise TypeError(f"Cannot convert {type(name)} to string.")


def feature_vector(fmap):
    return [(as_str(key), float(fmap[key])) for key in fmap]


class Dataset:
    def __init__(self, data, categorical=False):
        self.categorical = bool(categorical)
        self.instances = []
        self.features = set()

        for features, response in data:
            fv = feature_vector(features)
            self.instances.append((dict(fv), response))
            for k, _ in fv:
                self.features.add(k)

    @property
    def num_features(self):
        return len(self.features)

    def __len__(self):
        return len(self.instances)

    def __iter__(self):
        for inst in self.instances:
            yield inst

    def __getitem__(self, i):
        return self.instances[i]
