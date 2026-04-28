def parse_feature_line(line):
    """Parse tab-delimited GFF/GTF feature core columns."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 9:
        raise ValueError("feature line must have at least 9 tab-separated fields")

    chrom = parts[0]
    method = parts[1]
    featuretype = parts[2]
    start = int(parts[3])
    stop = int(parts[4])
    try:
        score = float(parts[5])
    except ValueError:
        score = 0.0
    strand = parts[6]
    phase = parts[7]
    raw_attributes = parts[8]

    return {
        "chrom": chrom,
        "method": method,
        "featuretype": featuretype,
        "start": start,
        "stop": stop,
        "score": score,
        "strand": strand,
        "phase": phase,
        "raw_attributes": raw_attributes,
    }


def parse_attributes(raw_attributes, field_sep):
    """Parse semicolon-separated attributes into a dict."""
    attrs = {}
    for item in raw_attributes.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        field, value = item.split(field_sep, 1)
        attrs[field.strip()] = value.replace('"', "").strip()
    return attrs


def parse_gff_attributes(raw_attributes):
    """Parse GFF attributes (field separator '=')."""
    return parse_attributes(raw_attributes, "=")


def parse_gtf_attributes(raw_attributes):
    """Parse GTF attributes (field separator space)."""
    return parse_attributes(raw_attributes, " ")


def add_attributes(raw_attributes, new_attrs, field_sep):
    """Append key/value attributes to raw attribute string."""
    pieces = []
    if raw_attributes and not raw_attributes.endswith(";"):
        raw_attributes = raw_attributes + ";"
    for key, value in new_attrs.items():
        pieces.append(f"{key}{field_sep}{value}")
    return (raw_attributes or "") + ";".join(pieces)
