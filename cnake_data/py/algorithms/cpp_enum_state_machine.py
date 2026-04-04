"""Finite state machine using IntEnum with dict-of-dicts transition table.

Keywords: algorithms, state machine, enum, FSM, transitions, benchmark
"""

from enum import IntEnum

from cnake_data.benchmarks import python_benchmark


class State(IntEnum):
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    ERROR = 3
    DONE = 4


# Transition table: TRANSITIONS[state][event] -> new_state or None (invalid)
# 5 states x 8 event types
TRANSITIONS: list[list[int | None]] = [
    # IDLE:    ev0=RUNNING, ev1=ERROR,   ev2=None,    ev3=None,    ev4=None,    ev5=DONE,    ev6=None,    ev7=IDLE
    [1, 3, None, None, None, 4, None, 0],
    # RUNNING: ev0=None,    ev1=PAUSED,  ev2=DONE,    ev3=ERROR,   ev4=RUNNING, ev5=None,    ev6=IDLE,    ev7=RUNNING
    [None, 2, 4, 3, 1, None, 0, 1],
    # PAUSED:  ev0=RUNNING, ev1=None,    ev2=IDLE,    ev3=ERROR,   ev4=None,    ev5=DONE,    ev6=None,    ev7=PAUSED
    [1, None, 0, 3, None, 4, None, 2],
    # ERROR:   ev0=IDLE,    ev1=IDLE,    ev2=None,    ev3=None,    ev4=None,    ev5=None,    ev6=IDLE,    ev7=RUNNING
    [0, 0, None, None, None, None, 0, 1],
    # DONE:    ev0=IDLE,    ev1=None,    ev2=None,    ev3=None,    ev4=None,    ev5=None,    ev6=None,    ev7=None
    [0, None, None, None, None, None, None, None],
]


@python_benchmark(args=(1000000,))
def cpp_enum_state_machine(n: int) -> tuple:
    """Process n events through a 5-state FSM, counting valid and invalid transitions.

    Event for step i: event = (i * 2654435761) % 8
    Initial state: IDLE (0). Valid transitions change state; invalid increment error counter.

    Args:
        n: Number of events to process.

    Returns:
        Tuple of (final_state_as_int, valid_transitions, invalid_transitions).
    """
    state = State.IDLE
    valid = 0
    invalid = 0
    for i in range(n):
        event = (i * 2654435761) % 8
        new_state = TRANSITIONS[state][event]
        if new_state is not None:
            state = State(new_state)
            valid += 1
        else:
            invalid += 1
    return (int(state), valid, invalid)
