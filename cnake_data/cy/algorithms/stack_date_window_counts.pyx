# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count date window ticks by year/month/day frequency rules (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: c3f770e30d5569e6492ce35ce6f794ded4ccf693
- filename: dateutils.pyx
"""

from cnake_data.benchmarks import cython_benchmark


cdef int _is_leap(int year) noexcept nogil:
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


cdef int _monthrange(int year, int month) noexcept nogil:
    cdef int mdays[13]
    mdays[0] = 0
    mdays[1] = 31
    mdays[2] = 28
    mdays[3] = 31
    mdays[4] = 30
    mdays[5] = 31
    mdays[6] = 30
    mdays[7] = 31
    mdays[8] = 31
    mdays[9] = 30
    mdays[10] = 31
    mdays[11] = 30
    mdays[12] = 31
    return mdays[month] + (1 if month == 2 and _is_leap(year) else 0)


@cython_benchmark(syntax="cy", args=(1993, 160, 2))
def stack_date_window_counts(int start_year, int span_years, int frequency):
    cdef int sy = start_year
    cdef int ey = start_year + span_years
    cdef int y = sy
    cdef int m = 1
    cdef int d = 1
    cdef int count = 0
    cdef int first = 0
    cdef int last = 0
    cdef unsigned int checksum = 0
    cdef int stamp

    while (y < ey) or (y == ey and (m < 12 or (m == 12 and d <= 31))):
        stamp = y * 10000 + m * 100 + d
        if count == 0:
            first = stamp
        last = stamp
        count += 1
        checksum = (checksum + <unsigned int>(stamp * (count + 13))) & 0xFFFFFFFF

        if frequency == 0:
            y += 1
            m = 1
            d = 1
        elif frequency == 1:
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1
            d = 1
        else:
            if d == _monthrange(y, m):
                d = 1
                if m == 12:
                    y += 1
                    m = 1
                else:
                    m += 1
            else:
                d += 1

    return (count, first, last, checksum)
