"""Count date window ticks by year/month/day frequency rules.

Adapted from The Stack v2 Cython candidate:
- blob_id: c3f770e30d5569e6492ce35ce6f794ded4ccf693
- filename: dateutils.pyx

Keywords: algorithms, date windows, leap year, frequency, iteration
"""

from cnake_data.benchmarks import python_benchmark

_MONTH_DAYS = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _is_leap(year: int) -> bool:
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def _monthrange(year: int, month: int) -> int:
    return _MONTH_DAYS[month] + (1 if month == 2 and _is_leap(year) else 0)


@python_benchmark(args=(1993, 160, 2))
def stack_date_window_counts(start_year: int, span_years: int, frequency: int) -> tuple:
    """Step through a date range and summarize count/checksums of visited dates."""
    sy, sm, sd = start_year, 1, 1
    ey, em, ed = start_year + span_years, 12, 31

    y, m, d = sy, sm, sd
    count = 0
    first = 0
    last = 0
    checksum = 0

    while (y, m, d) <= (ey, em, ed):
        stamp = y * 10000 + m * 100 + d
        if count == 0:
            first = stamp
        last = stamp
        count += 1
        checksum = (checksum + stamp * (count + 13)) & 0xFFFFFFFF

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
