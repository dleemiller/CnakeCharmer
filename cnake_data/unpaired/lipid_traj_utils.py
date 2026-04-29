"""Trajectory processing helpers for lipid/cholesterol bead coordinates."""

from __future__ import annotations

import re

import numpy as np


def chol_conc(filename):
    with open(filename) as file:
        for line in file:
            if re.match("CHOL", line):
                return int(line.split()[1]) * 2
    return 0


def process_traj(traj_filename, nchol, ndim, nconf):
    with open(traj_filename) as traj_file:
        n = int(traj_file.readline().split()[0])

    nperlipid = 12
    nperchol = 8
    nlipids = (n - nperchol * nchol) // nperlipid
    ncholbeads = nchol * nperchol
    nlipidbeads = nlipids * nperlipid

    l = np.zeros((ndim, nconf), dtype=float)
    x = np.zeros((nlipidbeads, ndim, nconf), dtype=float)
    y = np.zeros((ncholbeads, ndim, nconf), dtype=float)

    with open(traj_filename) as traj_file:
        for t in range(nconf):
            traj_file.readline()
            for k in range(ndim):
                l[k, t] = float(traj_file.readline().strip())

            for i in range(nlipidbeads // 2):
                line = traj_file.readline().split()
                for k in range(ndim):
                    x[i, k, t] = float(line[k + 1])

            for i in range(ncholbeads // 2):
                line = traj_file.readline().split()
                for k in range(ndim):
                    y[i, k, t] = float(line[k + 1])

            for i in range(nlipidbeads // 2):
                line = traj_file.readline().split()
                index = i + nlipidbeads // 2
                for k in range(ndim):
                    x[index, k, t] = float(line[k + 1])

            for i in range(ncholbeads // 2):
                line = traj_file.readline().split()
                index = i + ncholbeads // 2
                for k in range(ndim):
                    y[index, k, t] = float(line[k + 1])

    return n, l, x, y
