import csv


def gen_chunks(reader, chunk_size=100):
    """Yield chunk_size slices from an iterable CSV reader."""
    chunk = []
    for i, line in enumerate(reader):
        if i % chunk_size == 0 and i > 0:
            yield chunk
            chunk = []
        chunk.append(line)
    if chunk:
        yield chunk


def csv_event_rows_to_dict(path, chunk_size=10000, delimiter="\t"):
    """Group rows into dict[key] -> list[(col1, col2)] from a delimited file."""
    data = {}
    with open(path, encoding="utf-8", newline="") as data_file:
        csv_reader = csv.reader(data_file, delimiter=delimiter)
        for chunk in gen_chunks(csv_reader, chunk_size=chunk_size):
            for row in chunk:
                key = row[0]
                val = (row[1], row[2])
                if key in data:
                    data[key].append(val)
                else:
                    data[key] = [val]
    return data
