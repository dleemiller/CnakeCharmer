import random
from copy import deepcopy
from math import exp

kb = 1


class MonteCarlo:
    def __init__(self, system, temperature, dx, target, seed=12345):
        self.system = system
        self.dx = dx
        self.target = target
        self.temperature = temperature
        self.beta = 1 / (kb * temperature)
        random.seed(seed)
        self.n_accept = 0

    def take_step(self):
        for idx in range(self.system.n):
            pe_old = self.system.calc_energy(id=idx)
            xyz_old = deepcopy(self.system.traj.xyz[0])
            self.system.displace_coords(idx, self.dx)
            pe_new = self.system.calc_energy(id=idx)

            if pe_new:
                delta_pe = pe_new - pe_old
                if delta_pe < 0:
                    self.n_accept += 1
                else:
                    rand_value = random.random()
                    if exp(-self.beta * delta_pe) > rand_value:
                        self.n_accept += 1
                    else:
                        self.system.traj.xyz[0] = xyz_old
            else:
                self.system.traj.xyz[0] = xyz_old

    def relax(self, steps, adjust_freq=100):
        for i in range(steps):
            self.take_step()
            self.system.check_nlist()

            if i != 0 and i % adjust_freq == 0:
                prob = self.n_accept / (adjust_freq * self.system.n)
                if prob < self.target:
                    self.dx /= 1.025
                elif prob > self.target:
                    self.dx *= 1.025

                if self.dx > self.system.skin / 2:
                    self.dx = self.system.skin / 2

                potential = self.system.calc_energy(id=0, total=True)
                print(
                    f"Relax: {i} of {steps}\tdx: {self.dx:.6f}\tprob: {prob:.6f}\ttarget_prob: {self.target}  PE: {potential / self.system.n:.5f}"
                )
