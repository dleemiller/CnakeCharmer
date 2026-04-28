BUCKET_PER_TEN = 3
MAX_BUCKET_SIZE = 59
POWER_TEN = [10**i for i in range(20)]


def get_log10_round_to_floor(element):
    power = 0
    while power < len(POWER_TEN) and element >= POWER_TEN[power]:
        power += 1
    return power - 1


class DataflowDistributionCounter:
    def __init__(self):
        self.min = (1 << 63) - 1
        self.max = 0
        self.count = 0
        self.sum = 0
        self.buckets = [0] * MAX_BUCKET_SIZE
        self.is_cythonized = True

    def _fast_calculate_bucket_index(self, element):
        if element == 0:
            return 0
        log10_floor = get_log10_round_to_floor(element)
        power_of_ten = POWER_TEN[log10_floor]
        if element < power_of_ten * 2:
            bucket_offset = 0
        elif element < power_of_ten * 5:
            bucket_offset = 1
        else:
            bucket_offset = 2
        return 1 + log10_floor * BUCKET_PER_TEN + bucket_offset

    def add_input(self, element):
        if element < 0:
            raise ValueError("Distribution counters support only non-negative value")
        self.min = min(self.min, element)
        self.max = max(self.max, element)
        self.count += 1
        self.sum += element
        idx = self._fast_calculate_bucket_index(element)
        self.buckets[idx] += 1

    def add_input_n(self, element, n):
        if element < 0:
            raise ValueError("Distribution counters support only non-negative value")
        self.min = min(self.min, element)
        self.max = max(self.max, element)
        self.count += n
        self.sum += element * n
        idx = self._fast_calculate_bucket_index(element)
        self.buckets[idx] += n

    def translate_to_histogram(self):
        first = 0
        last = 0
        for i, v in enumerate(self.buckets):
            if v != 0:
                first = i
                break
        for i in range(len(self.buckets) - 1, -1, -1):
            if self.buckets[i] != 0:
                last = i
                break
        return {"firstBucketOffset": first, "bucketCounts": self.buckets[first : last + 1]}

    def extract_output(self):
        mean = self.sum // self.count if self.count else float("nan")
        return mean, self.sum, self.count, self.min, self.max
