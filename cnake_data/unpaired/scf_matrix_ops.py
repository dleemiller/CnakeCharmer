"""SCF helper kernels for density/fock/energy updates."""

from __future__ import annotations

import numpy as np


def make_density(no_of_electrons, c):
    nbasis = c.shape[0]
    d = np.zeros((nbasis, nbasis), dtype=float)
    occ = int(no_of_electrons / 2)
    for i in range(nbasis):
        for j in range(nbasis):
            for m in range(occ):
                d[i, j] += c[i, m] * c[j, m]
    return d


def make_fock(d, hamil, eri):
    n_basis = d.shape[0]
    fock = np.zeros((n_basis, n_basis), dtype=float)
    for i in range(n_basis):
        for j in range(n_basis):
            fock[i, j] = hamil[i, j]
            for k in range(n_basis):
                for l in range(n_basis):
                    fock[i, j] += d[k, l] * (2.0 * eri[i, j, k, l] - eri[i, k, j, l])
    return fock


def scf_energy(p, hcore, f):
    n = p.shape[0]
    e = 0.0
    for i in range(n):
        for j in range(n):
            e += p[i, j] * (hcore[i, j] + f[i, j])
    return e
