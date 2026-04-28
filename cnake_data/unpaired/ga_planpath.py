import numpy as np
from kinematic import forward_kinematic

FAILURE = 9487945.0


class build_path:
    def __init__(self, mechanismParams):
        self.targetPoint = mechanismParams["Target"]
        self.posture = mechanismParams["Posture"]
        self.POINTS = len(self.targetPoint)
        self.startstate = mechanismParams["startstate"]
        self.upper = [165, 125, 185, 190, 115, 360] * self.POINTS
        self.lower = [-165, -85, -55, -190, -115, -360] * self.POINTS

    def get_upper(self):
        return self.upper

    def get_lower(self):
        return self.lower

    def get_nParm(self):
        return 6 * self.POINTS

    def __call__(self, v):
        return self.run(v)

    def run(self, v):
        fitness = 0.0
        tmp_array = np.array(v).reshape((-1, 6))
        anglefitness = 0.0
        anglefit = 0.0

        for i, angles in enumerate(tmp_array):
            if i == 0:
                anglefit = np.sum(np.abs(self.startstate - tmp_array[i]))
            if i != 0:
                anglefitness = np.sum(np.abs(angles - tmp_array[i - 1]))
            x, y, z, posture_old = forward_kinematic(angles)
            x1, y1, z1 = self.targetPoint[i]
            distance = np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
            fitness += 50 * distance + 10 * anglefit + anglefitness

        return fitness

    def get_coordinates(self, v):
        return np.array(v).reshape((-1, 6))
