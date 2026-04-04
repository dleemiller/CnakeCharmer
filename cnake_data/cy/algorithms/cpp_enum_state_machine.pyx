# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# distutils: language = c++
"""Finite state machine using a C++ enum class for state representation.

Keywords: algorithms, state machine, enum class, FSM, transitions, cppclass, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

cdef extern from *:
    """
    enum class State : int {
        IDLE    = 0,
        RUNNING = 1,
        PAUSED  = 2,
        ERROR   = 3,
        DONE    = 4
    };
    """
    cpdef enum class State(int):
        IDLE    = 0
        RUNNING = 1
        PAUSED  = 2
        ERROR   = 3
        DONE    = 4

# Transition table: table[state * 8 + event] = new_state, -1 for invalid
# Encoded flat from the 5x8 transition matrix:
#   IDLE:    [1,3,-1,-1,-1,4,-1,0]
#   RUNNING: [-1,2,4,3,1,-1,0,1]
#   PAUSED:  [1,-1,0,3,-1,4,-1,2]
#   ERROR:   [0,0,-1,-1,-1,-1,0,1]
#   DONE:    [0,-1,-1,-1,-1,-1,-1,-1]
cdef int TRANS[40]

TRANS[0]  = 1;  TRANS[1]  = 3;  TRANS[2]  = -1; TRANS[3]  = -1
TRANS[4]  = -1; TRANS[5]  = 4;  TRANS[6]  = -1; TRANS[7]  = 0
TRANS[8]  = -1; TRANS[9]  = 2;  TRANS[10] = 4;  TRANS[11] = 3
TRANS[12] = 1;  TRANS[13] = -1; TRANS[14] = 0;  TRANS[15] = 1
TRANS[16] = 1;  TRANS[17] = -1; TRANS[18] = 0;  TRANS[19] = 3
TRANS[20] = -1; TRANS[21] = 4;  TRANS[22] = -1; TRANS[23] = 2
TRANS[24] = 0;  TRANS[25] = 0;  TRANS[26] = -1; TRANS[27] = -1
TRANS[28] = -1; TRANS[29] = -1; TRANS[30] = 0;  TRANS[31] = 1
TRANS[32] = 0;  TRANS[33] = -1; TRANS[34] = -1; TRANS[35] = -1
TRANS[36] = -1; TRANS[37] = -1; TRANS[38] = -1; TRANS[39] = -1


@cython_benchmark(syntax="cy", args=(1000000,))
def cpp_enum_state_machine(int n):
    """Process n events through a 5-state C++ enum class FSM.

    Event for step i: event = (i * 2654435761) % 8
    Initial state: IDLE (0). Valid transitions change state; invalid increment error counter.

    Args:
        n: Number of events to process.

    Returns:
        Tuple of (final_state_as_int, valid_transitions, invalid_transitions).
    """
    cdef State state = State.IDLE
    cdef int valid = 0
    cdef int invalid = 0
    cdef int i, event, new_state_int, idx

    for i in range(n):
        event = <int>((<long long>i * <long long>2654435761) % 8)
        idx = (<int>state) * 8 + event
        new_state_int = TRANS[idx]
        if new_state_int != -1:
            state = <State>new_state_int
            valid += 1
        else:
            invalid += 1

    return (<int>state, valid, invalid)
