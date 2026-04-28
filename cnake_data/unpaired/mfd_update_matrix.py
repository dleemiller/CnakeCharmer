def update_m_fast(m_coo, multipliers, m_data_for_update, m_e_locations, number_of_cells):
    for cell_index in range(number_of_cells):
        m_start = m_e_locations[cell_index]
        m_end = m_e_locations[cell_index + 1]

        for local_index in range(m_start, m_end):
            m_coo[local_index] = m_data_for_update[local_index] * multipliers[cell_index]
