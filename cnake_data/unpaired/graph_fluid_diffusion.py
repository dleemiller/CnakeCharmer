def graph_fluid_diffusion(indptr, indices, data, scores, fluid, damping_factor, n_iter, tol):
    """Diffuse fluid through a CSC/CSR-like graph structure."""
    n = len(fluid)
    restart_prob = 1.0 - damping_factor
    residu = restart_prob

    for _ in range(n_iter):
        for i in range(n):
            sent = fluid[i]
            if sent > 0.0:
                scores[i] += sent
                fluid[i] = 0.0

                j1 = indptr[i]
                j2 = indptr[i + 1]
                tmp = sent * damping_factor

                if j2 != j1:
                    for jj in range(j1, j2):
                        j = indices[jj]
                        fluid[j] += tmp * data[jj]
                    removed = sent * restart_prob
                else:
                    removed = sent

                residu -= removed

        if residu < tol * restart_prob:
            return


def diffusion_iteration_copy(indptr, indices, data, scores, fluid, damping_factor, n_iter, tol):
    """Functional wrapper returning updated scores/fluid arrays."""
    new_scores = list(scores)
    new_fluid = list(fluid)
    graph_fluid_diffusion(
        indptr,
        indices,
        data,
        new_scores,
        new_fluid,
        damping_factor,
        n_iter,
        tol,
    )
    return new_scores, new_fluid
