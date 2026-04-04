"""1D reaction-diffusion simulation (Gray-Scott model).

Keywords: simulation, reaction-diffusion, gray-scott, PDE, pattern formation, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def reaction_diffusion(n: int) -> float:
    """Simulate 1D Gray-Scott reaction-diffusion on n cells for 1000 steps.

    Two species U and V with:
      dU/dt = Du * laplacian(U) - U*V^2 + F*(1-U)
      dV/dt = Dv * laplacian(V) + U*V^2 - (F+k)*V

    Parameters: Du=0.16, Dv=0.08, F=0.035, k=0.065, dt=1.0.
    Initial: U[i] = 1 - exp(-((i - n/2)^2)/(n/10)^2)
             V[i] = exp(-((i - n/2)^2)/(n/10)^2)
    Periodic boundary conditions.

    Args:
        n: Number of cells.

    Returns:
        Sum of V values in the final state.
    """
    steps = 1000
    du_coeff = 0.16
    dv_coeff = 0.08
    feed = 0.035
    kill = 0.065
    dt = 1.0
    sigma = n / 10.0

    # Initialize
    u_arr = [0.0] * n
    v_arr = [0.0] * n
    for i in range(n):
        dist_sq = (i - n / 2.0) ** 2
        gauss = math.exp(-dist_sq / (sigma * sigma))
        u_arr[i] = 1.0 - gauss
        v_arr[i] = gauss

    u_new = [0.0] * n
    v_new = [0.0] * n

    for _t in range(steps):
        for i in range(n):
            im1 = (i - 1) % n
            ip1 = (i + 1) % n
            lap_u = u_arr[im1] - 2.0 * u_arr[i] + u_arr[ip1]
            lap_v = v_arr[im1] - 2.0 * v_arr[i] + v_arr[ip1]
            uvv = u_arr[i] * v_arr[i] * v_arr[i]
            u_new[i] = u_arr[i] + dt * (du_coeff * lap_u - uvv + feed * (1.0 - u_arr[i]))
            v_new[i] = v_arr[i] + dt * (dv_coeff * lap_v + uvv - (feed + kill) * v_arr[i])
        u_arr, u_new = u_new, u_arr
        v_arr, v_new = v_new, v_arr

    total = 0.0
    for i in range(n):
        total += v_arr[i]
    return total
