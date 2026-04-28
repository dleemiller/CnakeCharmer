MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
MULT = 6364136223846793005


def _uint32(n):
    return n & MASK32


def _uint64(n):
    return n & MASK64


def rng(state=0, inc=0):
    return [_uint64(state), _uint64(inc)]


def random_r(rng_state):
    """Advance PCG32 state and return next uint32."""
    oldstate = rng_state[0]
    rng_state[0] = _uint64(oldstate * MULT + rng_state[1])
    xorshifted = _uint32(((oldstate >> 18) ^ oldstate) >> 27)
    rot = _uint32(oldstate >> 59)
    return _uint32((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))


def srandom_r(rng_state, initstate, initseq):
    initstate = _uint64(initstate)
    initseq = _uint64(initseq)

    rng_state[0] = 0
    rng_state[1] = _uint64(initseq << 1) | 1
    random_r(rng_state)
    rng_state[0] = _uint64(rng_state[0] + initstate)
    random_r(rng_state)


def boundedrand_r(rng_state, bound):
    threshold = (-bound) % bound
    while True:
        r = random_r(rng_state)
        if r >= threshold:
            return r % bound


def unit_interval_r(rng_state):
    return float(random_r(rng_state)) / MASK32


def random_r_arr(rng_state, n):
    return [random_r(rng_state) for _ in range(n)]


def boundedrand_r_arr(rng_state, bound, n):
    return [boundedrand_r(rng_state, bound) for _ in range(n)]
