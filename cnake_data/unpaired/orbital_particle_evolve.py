import math


def c_evolve(r_i, ang_speed_i, timestep, nsteps):
    """In-place circular-motion update for particle positions."""
    nparticles = len(r_i)

    for _ in range(nsteps):
        for j in range(nparticles):
            x = r_i[j][0]
            y = r_i[j][1]
            ang_speed = ang_speed_i[j]

            norm = math.sqrt(x * x + y * y)
            if norm == 0.0:
                continue

            vx = -y / norm
            vy = x / norm

            dx = timestep * ang_speed * vx
            dy = timestep * ang_speed * vy

            r_i[j][0] += dx
            r_i[j][1] += dy

    return r_i
