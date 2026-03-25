#!/usr/bin/env python
"""
Minimal end-to-end training test with TRL GRPOTrainer.

Runs 1 training step with a few problems using environment_factory
for multi-turn Cython tool calling.

Usage:
    uv run python scripts/test_training.py
"""

import logging

from cnake_charmer.sources.base import ProblemSpec
from cnake_charmer.training.grpo import create_trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

# Test problems
problems = [
    ProblemSpec(
        problem_id="test_fib",
        description="Compute Fibonacci numbers up to a limit",
        python_code=(
            "def fibonacci(int_limit):\n"
            "    result = []\n"
            "    a, b = 0, 1\n"
            "    while a < int_limit:\n"
            "        result.append(a)\n"
            "        a, b = b, a + b\n"
            "    return result\n"
        ),
        func_name="fibonacci",
        test_cases=[((100,),), ((1000,),)],
        benchmark_args=(100000,),
        category="numerical",
        difficulty="easy",
        source="manual",
    ),
    ProblemSpec(
        problem_id="test_primes",
        description="Generate first N prime numbers",
        python_code=(
            "def primes(nb_primes):\n"
            "    primes_list = []\n"
            "    n = 2\n"
            "    while len(primes_list) < nb_primes:\n"
            "        for prime in primes_list:\n"
            "            if n % prime == 0:\n"
            "                break\n"
            "        else:\n"
            "            primes_list.append(n)\n"
            "        n += 1\n"
            "    return primes_list\n"
        ),
        func_name="primes",
        test_cases=[((10,),), ((20,),)],
        benchmark_args=(100,),
        category="numerical",
        difficulty="easy",
        source="manual",
    ),
]

trainer = create_trainer(
    model="Qwen/Qwen3-0.6B",
    problems=problems,
    output_dir="./output/test_run",
    num_generations=2,
    max_completion_length=1024,
    max_tool_calling_iterations=2,
    use_vllm=False,
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=100,
    report_to="none",
    peft_config={"r": 16, "lora_alpha": 32, "target_modules": "all-linear"},
)

trainer.train()
