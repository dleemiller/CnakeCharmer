"""Genetic algorithm minimising the Rastrigin function using a deterministic LCG.

Keywords: optimization, genetic algorithm, rastrigin, evolutionary, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark

_POP = 200
_GENES = 16
_LO = -5.12
_HI = 5.12
_A = 10.0
_MUT_RATE = 0.1  # per-gene mutation probability
_MUT_SCALE = 0.5  # mutation delta scale
_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_MOD = 0x100000000  # 2^32


def _rastrigin(genes: list) -> float:
    acc = _A * len(genes)
    for x in genes:
        acc += x * x - _A * math.cos(2.0 * math.pi * x)
    return acc


@python_benchmark(args=(500,))
def genetic_algorithm(n: int) -> tuple:
    """Minimise Rastrigin (16-dim) via GA for n generations.

    Population: 200 individuals.  Uses a 32-bit LCG for all randomness so
    Python and Cython produce identical floating-point sequences.

    Args:
        n: Number of generations.

    Returns:
        Tuple of (best_fitness_ever, mean_fitness_final, best_fitness_gen_half).
    """
    TWO_PI = 2.0 * math.pi  # noqa: F841
    state = 98765  # LCG seed

    def lcg_float() -> float:
        nonlocal state
        state = (state * _LCG_A + _LCG_C) & 0xFFFFFFFF
        return state / 4294967296.0

    # Initialise population
    pop = [[_LO + lcg_float() * (_HI - _LO) for _ in range(_GENES)] for _ in range(_POP)]
    fitness = [_rastrigin(ind) for ind in pop]

    best_ever = min(fitness)
    best_half = best_ever  # will be overwritten at generation n//2
    half_gen = n >> 1

    for gen in range(n):
        # Tournament selection + crossover → new population
        new_pop = []
        for _ in range(_POP):
            # Select two parents via 2-tournament
            ia = int(lcg_float() * _POP)
            ib = int(lcg_float() * _POP)
            ic = int(lcg_float() * _POP)
            id_ = int(lcg_float() * _POP)
            p1 = ia if fitness[ia] < fitness[ib] else ib
            p2 = ic if fitness[ic] < fitness[id_] else id_

            # Single-point crossover
            cut = int(lcg_float() * (_GENES + 1))
            child = pop[p1][:cut] + pop[p2][cut:]

            # Per-gene mutation
            for g in range(_GENES):
                if lcg_float() < _MUT_RATE:
                    delta = (lcg_float() - 0.5) * 2.0 * _MUT_SCALE
                    v = child[g] + delta
                    if v < _LO:
                        v = _LO
                    elif v > _HI:
                        v = _HI
                    child[g] = v
            new_pop.append(child)

        pop = new_pop
        fitness = [_rastrigin(ind) for ind in pop]

        gen_best = min(fitness)
        if gen_best < best_ever:
            best_ever = gen_best

        if gen == half_gen:
            best_half = gen_best

    mean_final = sum(fitness) / _POP
    return (best_ever, mean_final, best_half)
