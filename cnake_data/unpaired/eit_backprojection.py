import numpy as np


class BackProjection:
    def __init__(self, mesh, forward):
        self.node = mesh["node"]
        self.element = mesh["element"]
        self.alpha = mesh["alpha"]
        self.jacobian = forward.jac
        self.B = forward.b_matrix
        self.H = self.B
        self.setup()

    def setup(self, weight="none"):
        if weight == "simple":
            weight_m = self.simpleWeight(self.B.shape[0])
            self.H = weight_m * self.B

    def solve(self, v1, v0, normalize=True):
        if normalize:
            vn = -(v1 - v0) / np.sign(v0)
        else:
            vn = v1 - v0
        hasil = np.dot(self.H.transpose(), vn)
        return np.real(hasil)

    def solveGramSchmidt(self, v1, v0):
        a = np.dot(v1, v0) / np.dot(v0, v0)
        vn = -(v1 - a * v0) / np.sign(v0)
        hasil = np.dot(self.H.transpose(), vn)
        return np.real(hasil)

    @property
    def nodeXY(self):
        return self.node[:, :2]

    def simpleWeight(self, numVoltages):
        d = np.sqrt(np.sum(self.nodeXY**2, axis=1))
        r = np.max(d)
        w = (1.01 * r - d) / (1.01 * r)
        weights = np.dot(np.ones((numVoltages, 1)), w.reshape(1, -1))
        return weights
