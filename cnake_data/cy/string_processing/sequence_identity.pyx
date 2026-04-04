# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Pairwise sequence identity computation over deterministic aligned sequence pairs (Cython-optimized).

Keywords: string processing, sequence identity, alignment, bioinformatics, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

# Alphabet as bytes: "ACDEFGHIKLMNPQRSTVWY-"
# A=65 C=67 D=68 E=69 F=70 G=71 H=72 I=73 K=75 L=76
# M=77 N=78 P=80 Q=81 R=82 S=83 T=84 V=86 W=87 Y=89 -=45
cdef unsigned char ALPHABET[21]
ALPHABET[:] = [65, 67, 68, 69, 70, 71, 72, 73, 75, 76,
               77, 78, 80, 81, 82, 83, 84, 86, 87, 89, 45]


@cython_benchmark(syntax="cy", args=(10000,))
def sequence_identity(int n):
    """Compute pairwise alignment identity for n sequence pairs.

    Args:
        n: Number of sequence pairs to evaluate.

    Returns:
        Tuple of (total_matches, pairs_above_half) as integers.
    """
    cdef int i, j, matches
    cdef long long total_matches = 0
    cdef int pairs_above_half = 0
    cdef unsigned char qc, tc
    cdef int seq_len = 100
    cdef unsigned char gap = 45  # '-'

    with nogil:
        for i in range(n):
            matches = 0
            for j in range(seq_len):
                qc = ALPHABET[(i * 7 + j * 13) % 21]
                tc = ALPHABET[(i * 11 + j * 17) % 21]
                if qc == tc and qc != gap:
                    matches += 1
            total_matches += matches
            if matches > 50:
                pairs_above_half += 1

    return (int(total_matches), pairs_above_half)
