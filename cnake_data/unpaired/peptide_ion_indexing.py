import math


def get_ions_mz_index(
    peptide_mass,
    prefix_mass,
    delta_m,
    mass_h2o,
    mass_nh3,
    mass_h,
    mass_co,
    mass_c_terminus,
):
    """Compute 18 ion m/z-bin indices for a peptide prefix split."""
    b_ion_mass = prefix_mass + mass_h
    a_ion_mass = b_ion_mass - mass_co
    y_ion_mass = peptide_mass - prefix_mass + mass_h + mass_c_terminus + mass_h

    b_ions = [b_ion_mass, b_ion_mass - mass_h2o, b_ion_mass - mass_nh3]
    a_ions = [a_ion_mass, a_ion_mass - mass_h2o, a_ion_mass - mass_nh3]
    y_ions = [y_ion_mass, y_ion_mass - mass_h2o, y_ion_mass - mass_nh3]

    charge1 = b_ions + a_ions + y_ions
    charge2 = [(mz + mass_h) / 2.0 for mz in charge1]
    ions_mz = charge1 + charge2

    return [int(math.floor(mz / delta_m)) for mz in ions_mz]


def process_spectrum(mz_list, intensity_list, m_size, resolution):
    """Project sparse (m/z, intensity) peaks into a normalized dense vector."""
    holder = [0.0 for _ in range(m_size)]

    for mz, intensity in zip(mz_list, intensity_list, strict=False):
        loc = int(round(mz * resolution))
        if 0 <= loc < m_size:
            holder[loc] += float(intensity)

    total = sum(holder)
    if total > 0.0:
        holder = [x / total for x in holder]
    return holder
