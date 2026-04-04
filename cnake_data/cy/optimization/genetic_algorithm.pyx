# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Genetic algorithm minimising the Rastrigin function (Cython-optimized).

Keywords: optimization, genetic algorithm, rastrigin, evolutionary, cython, benchmark
"""

from libc.math cimport cos
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark

cdef int   _POP       = 200
cdef int   _GENES     = 16
cdef double _LO       = -5.12
cdef double _HI       = 5.12
cdef double _A        = 10.0
cdef double _MUT_RATE = 0.1
cdef double _MUT_SCALE = 0.5
cdef unsigned int _LCG_A = 1664525
cdef unsigned int _LCG_C = 1013904223


cdef inline double _rastrigin(double *ind, int genes) noexcept nogil:
    cdef double acc = _A * genes
    cdef int g
    cdef double x
    for g in range(genes):
        x = ind[g]
        acc += x * x - _A * cos(2.0 * 3.141592653589793 * x)
    return acc


@cython_benchmark(syntax="cy", args=(500,))
def genetic_algorithm(int n):
    """Minimise Rastrigin (16-dim) via GA for n generations.

    Returns:
        Tuple of (best_fitness_ever, mean_fitness_final, best_fitness_gen_half).
    """
    cdef int POP   = _POP
    cdef int GENES = _GENES
    cdef int half_gen = n >> 1

    # Allocate: pop[POP * GENES], new_pop[POP * GENES], fitness[POP]
    cdef double *pop     = <double *>malloc(POP * GENES * sizeof(double))
    cdef double *new_pop = <double *>malloc(POP * GENES * sizeof(double))
    cdef double *fitness = <double *>malloc(POP * sizeof(double))
    if pop == NULL or new_pop == NULL or fitness == NULL:
        if pop != NULL: free(pop)
        if new_pop != NULL: free(new_pop)
        if fitness != NULL: free(fitness)
        raise MemoryError()

    cdef unsigned int state = 98765  # same seed as Python
    cdef int i, g, gen, ia, ib, ic, id_, p1, p2, cut
    cdef double best_ever, best_half, mean_final, gen_best, delta, v
    cdef double *child
    cdef double *parent1
    cdef double *parent2
    cdef double *tmp_ptr

    with nogil:
        # Initialise population
        for i in range(POP):
            for g in range(GENES):
                state = state * _LCG_A + _LCG_C
                pop[i * GENES + g] = _LO + (<double>state / 4294967296.0) * (_HI - _LO)

        # Initial fitness
        for i in range(POP):
            fitness[i] = _rastrigin(pop + i * GENES, GENES)

        best_ever = fitness[0]
        for i in range(1, POP):
            if fitness[i] < best_ever:
                best_ever = fitness[i]
        best_half = best_ever

        for gen in range(n):
            # Tournament selection + crossover → new_pop
            for i in range(POP):
                # Two 2-tournaments
                state = state * _LCG_A + _LCG_C
                ia = <int>((<double>state / 4294967296.0) * POP)
                state = state * _LCG_A + _LCG_C
                ib = <int>((<double>state / 4294967296.0) * POP)
                state = state * _LCG_A + _LCG_C
                ic = <int>((<double>state / 4294967296.0) * POP)
                state = state * _LCG_A + _LCG_C
                id_ = <int>((<double>state / 4294967296.0) * POP)
                p1 = ia if fitness[ia] < fitness[ib] else ib
                p2 = ic if fitness[ic] < fitness[id_] else id_

                # Single-point crossover
                state = state * _LCG_A + _LCG_C
                cut = <int>((<double>state / 4294967296.0) * (GENES + 1))
                parent1 = pop + p1 * GENES
                parent2 = pop + p2 * GENES
                child = new_pop + i * GENES
                for g in range(cut):
                    child[g] = parent1[g]
                for g in range(cut, GENES):
                    child[g] = parent2[g]

                # Per-gene mutation
                for g in range(GENES):
                    state = state * _LCG_A + _LCG_C
                    if (<double>state / 4294967296.0) < _MUT_RATE:
                        state = state * _LCG_A + _LCG_C
                        delta = ((<double>state / 4294967296.0) - 0.5) * 2.0 * _MUT_SCALE
                        v = child[g] + delta
                        if v < _LO:
                            v = _LO
                        elif v > _HI:
                            v = _HI
                        child[g] = v

            # Swap populations
            tmp_ptr = pop
            pop = new_pop
            new_pop = tmp_ptr

            # Re-evaluate fitness
            for i in range(POP):
                fitness[i] = _rastrigin(pop + i * GENES, GENES)

            gen_best = fitness[0]
            for i in range(1, POP):
                if fitness[i] < gen_best:
                    gen_best = fitness[i]
            if gen_best < best_ever:
                best_ever = gen_best
            if gen == half_gen:
                best_half = gen_best

        # Final mean
        mean_final = 0.0
        for i in range(POP):
            mean_final += fitness[i]
        mean_final /= POP

    free(pop)
    free(new_pop)
    free(fitness)
    return (best_ever, mean_final, best_half)
