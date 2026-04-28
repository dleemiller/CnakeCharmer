import numpy as np
from sklearn.neighbors import KDTree

DEG_TO_RAD = np.pi / 180.0


def rotate(x, y, costheta, sintheta):
    xt = x * costheta - y * sintheta
    yt = x * sintheta + y * costheta
    return xt, yt


class QueryCat:
    def __init__(self, xy, scale_x=1.0, scale_y=1.0, angle=0.0):
        norm = np.sqrt(scale_x * scale_x / 2.0 + scale_y * scale_y / 2.0)
        self.scale_x = scale_x / norm
        self.scale_y = scale_y / norm
        self.sintheta = np.sin(angle * DEG_TO_RAD)
        self.costheta = np.cos(angle * DEG_TO_RAD)
        self.xy = np.asarray(xy, dtype=float)
        self.xyt = self.transform(self.xy)
        self._lookup_tree = KDTree(self.xyt)

    def transform(self, xy):
        out = np.empty_like(xy, dtype=float)
        for i in range(xy.shape[0]):
            x, y = rotate(xy[i, 0], xy[i, 1], self.costheta, self.sintheta)
            out[i, 0] = x * self.scale_x
            out[i, 1] = y * self.scale_y
        return out

    def query_disk(self, centers, radius):
        centers_t = self.transform(np.asarray(centers, dtype=float))
        return self._lookup_tree.query_radius(centers_t, radius)
