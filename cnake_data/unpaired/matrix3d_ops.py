"""3x3 matrix operations and vector dot products."""

from __future__ import annotations


class Matrix3D:
    def __init__(self, row1, row2, row3):
        self.data = [list(row1), list(row2), list(row3)]

    def __iter__(self):
        for r in self.data:
            yield tuple(r)

    def determinant(self):
        m = self.data
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    def inverse(self):
        m = self.data
        invdet = 1.0 / self.determinant()

        minv = Matrix3D((0, 0, 0), (0, 0, 0), (0, 0, 0))
        minv.data[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet
        minv.data[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet
        minv.data[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet
        minv.data[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet
        minv.data[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet
        minv.data[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet
        minv.data[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet
        minv.data[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet
        minv.data[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet
        return minv

    def dot_vec(self, vec):
        m = self.data
        return (
            m[0][0] * vec[0] + m[0][1] * vec[1] + m[0][2] * vec[2],
            m[1][0] * vec[0] + m[1][1] * vec[1] + m[1][2] * vec[2],
            m[2][0] * vec[0] + m[2][1] * vec[1] + m[2][2] * vec[2],
        )

    def transpose(self):
        m = self.data
        return Matrix3D(
            (m[0][0], m[1][0], m[2][0]), (m[0][1], m[1][1], m[2][1]), (m[0][2], m[1][2], m[2][2])
        )

    @classmethod
    def identity(cls, diag=1.0):
        return cls((diag, 0.0, 0.0), (0.0, diag, 0.0), (0.0, 0.0, diag))

    @classmethod
    def zeros(cls):
        return cls.identity(diag=0.0)
