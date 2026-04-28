INTEGER = "Integer"
FLOAT = "Float"
NUMERIC = "Numeric"


def _map(func, iterable, bad=None):
    if bad is None:
        bad = [".", ""]
    return [func(x) if x not in bad else None for x in iterable]


def _parse_filter(filt_str):
    if filt_str == ".":
        return None
    if filt_str == "PASS":
        return []
    return filt_str.split(";")


def parse_samples(names, samples, samp_fmt, samp_fmt_types, samp_fmt_nums, site, call_cls=None):
    if call_cls is None:
        call_cls = dict

    samp_data = []
    n_samples = len(samples)
    n_formats = len(samp_fmt._fields)

    for i in range(n_samples):
        name = names[i]
        sample = samples[i]
        sampdat = [None] * n_formats
        sampvals = sample.split(":")

        for j in range(n_formats):
            if j >= len(sampvals):
                break
            vals = sampvals[j]

            if samp_fmt._fields[j] == "GT":
                sampdat[j] = vals
                continue
            if samp_fmt._fields[j] == "FT":
                sampdat[j] = _parse_filter(vals)
                continue
            if not vals or vals == ".":
                sampdat[j] = None
                continue

            entry_type = samp_fmt_types[j]
            entry_num = samp_fmt_nums[j]

            if entry_num == 1:
                if entry_type == INTEGER:
                    try:
                        sampdat[j] = int(vals)
                    except ValueError:
                        sampdat[j] = float(vals)
                elif entry_type in (FLOAT, NUMERIC):
                    sampdat[j] = float(vals)
                else:
                    sampdat[j] = vals
                continue

            vals_split = vals.split(",")
            if entry_type == INTEGER:
                try:
                    sampdat[j] = _map(int, vals_split)
                except ValueError:
                    sampdat[j] = _map(float, vals_split)
            elif entry_type in (FLOAT, NUMERIC):
                sampdat[j] = _map(float, vals_split)
            else:
                sampdat[j] = _map(str, vals_split)

        sampdict = dict(zip(samp_fmt._fields, sampdat, strict=False))
        samp_data.append(call_cls(site=site, sample=name, data=sampdict))

    return samp_data
