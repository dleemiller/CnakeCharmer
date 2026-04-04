"""Evaluate reverse Polish notation expressions.

Keywords: rpn, stack, calculator, expression evaluation, algorithms, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def rpn_eval(n: int) -> float:
    """Generate and evaluate n RPN operations, returning the final accumulator.

    Generates a deterministic stream of operands and operators (+, -, *, /),
    evaluates them using a stack-based RPN interpreter, and accumulates results.

    Args:
        n: Number of RPN operations to evaluate.

    Returns:
        Sum of all expression results.
    """
    # Token types
    OP_ADD = 1
    OP_SUB = 2
    OP_MUL = 3
    OP_DIV = 4

    stack = [0.0] * 1024
    sp = 0  # stack pointer
    accumulator = 0.0

    for i in range(n):
        h = ((i * 2654435761 + 1013904223) >> 8) & 0xFFFF

        if sp < 2 or h % 5 == 0:
            # Push a value
            val = ((h * 31 + 7) % 200 - 100) / 10.0
            stack[sp] = val
            sp += 1
            if sp >= 1024:
                sp = 1
        else:
            op = h % 4 + 1  # 1-4
            b = stack[sp - 1]
            a = stack[sp - 2]
            sp -= 2

            if op == OP_ADD:
                result = a + b
            elif op == OP_SUB:
                result = a - b
            elif op == OP_MUL:
                result = a * b
            elif op == OP_DIV:
                result = a / b if abs(b) > 1e-10 else 0.0
            else:
                result = 0.0

            stack[sp] = result
            sp += 1
            accumulator += result

    return accumulator
