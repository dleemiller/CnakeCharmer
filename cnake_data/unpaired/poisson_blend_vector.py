def create_vector(source, reference, mask, multiplier=1.0):
    """Construct RHS vector for Poisson blending preserving source gradient."""
    height = len(mask)
    width = len(mask[0]) if height else 0
    out = []

    for j in range(height):
        for i in range(width):
            if mask[j][i] == 0:
                continue

            neighbors = 0
            coeff = 0.0
            s = multiplier * source[j][i]

            nbrs = [(j - 1, i), (j + 1, i), (j, i + 1), (j, i - 1)]
            for nj, ni in nbrs:
                if mask[nj][ni] == 0:
                    coeff += 2.0 * reference[nj][ni]
                    coeff -= 2.0 * multiplier * source[nj][ni]
                else:
                    neighbors += 1
                    coeff -= 4.0 * multiplier * source[nj][ni]

            coeff += (2 * neighbors + 8) * s
            out.append(coeff)

    return out


def create_vector_from_field(source_field, reference, mask):
    """Construct RHS vector from precomputed divergence/field term."""
    height = len(mask)
    width = len(mask[0]) if height else 0
    out = []

    for j in range(height):
        for i in range(width):
            if mask[j][i] == 0:
                continue

            neighbors = 0
            coeff = 0.0
            s = source_field[j][i]

            for nj, ni in ((j - 1, i), (j + 1, i), (j, i + 1), (j, i - 1)):
                if mask[nj][ni] == 0:
                    coeff += 2.0 * reference[nj][ni]
                else:
                    neighbors += 1

            coeff += neighbors * s
            out.append(coeff)

    return out
