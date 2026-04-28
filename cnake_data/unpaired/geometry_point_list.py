from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    x: float
    y: float
    z: float


class PointList:
    def __init__(self, xyz=None):
        if xyz is None:
            self._data = np.empty((3, 0), dtype=float)
        elif isinstance(xyz, np.ndarray):
            if xyz.shape[1] == 3:
                self._data = xyz.transpose()
            elif xyz.shape[0] == 3:
                self._data = xyz
            else:
                raise ValueError("Wrong shape.")
        elif isinstance(xyz, list):
            l = len(xyz)
            self._data = np.empty((3, l), dtype=float)
            for i in range(l):
                self._data[:, i] = (xyz[i].x, xyz[i].y, xyz[i].z)
        else:
            raise ValueError("Wrong input")

    def __len__(self):
        return self._data.shape[1]

    def __getitem__(self, key):
        return Point(self._data[0, key], self._data[1, key], self._data[2, key])

    def __setitem__(self, key, value):
        self._data[:, key] = (value.x, value.y, value.z)

    def __iter__(self):
        for i in range(self._data.shape[1]):
            yield self[i]

    def __repr__(self):
        return "Points"

    def __eq__(self, other):
        return isinstance(other, PointList) and np.all(self._data == other._data)

    @property
    def x(self):
        return self._data[0]

    @property
    def y(self):
        return self._data[1]

    @property
    def z(self):
        return self._data[2]

    def as_array(self):
        return self._data.transpose()
