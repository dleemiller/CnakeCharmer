"""
Minimal memory smoke test for GRPO training.

Tests that gpt-oss-20b-cython-sft-merged + LoRA + vLLM colocate fits in GPU memory.
Runs a single training step with 1 problem and 2 generations.

Usage:
    uv run --no-sync python scripts/test_grpo_memory.py
    uv run --no-sync python scripts/test_grpo_memory.py --use-vllm     # test with vLLM colocate
    uv run --no-sync python scripts/test_grpo_memory.py --no-tools     # skip tool execution, just test model memory
"""

import argparse
import gc
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def print_gpu_memory(label: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - reserved
        logger.info(
            f"[GPU {label}] Allocated: {allocated:.1f}GB | Reserved: {reserved:.1f}GB | "
            f"Free: {free:.1f}GB | Total: {total:.1f}GB"
        )


def test_model_loading(model_path: str, use_lora: bool = True):
    """Test 1: Can we load the model + LoRA?"""
    logger.info("=" * 60)
    logger.info("TEST 1: Model Loading")
    logger.info("=" * 60)
    print_gpu_memory("before load")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print_gpu_memory("after model load")

    if use_lora:
        from peft import LoraConfig, get_peft_model

        logger.info("Applying LoRA (rank=32)...")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules="all-linear",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print_gpu_memory("after LoRA")

    return model, tokenizer


def test_generation(model, tokenizer):
    """Test 2: Can we generate a completion?"""
    logger.info("=" * 60)
    logger.info("TEST 2: Generation")
    logger.info("=" * 60)

    prompt = "Translate this Python code to optimized Cython:\n```python\ndef sum_array(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total\n```"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    logger.info("Generating 128 tokens...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )
    print_gpu_memory("after generation")

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    logger.info(f"Generated: {generated[:200]}...")


def test_grpo_trainer(model_path: str, use_vllm: bool = False, no_tools: bool = False):
    """Test 3: Can we create and step a GRPOTrainer?"""
    logger.info("=" * 60)
    logger.info("TEST 3: GRPOTrainer Single Step")
    logger.info("=" * 60)
    print_gpu_memory("before trainer")

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    # Minimal dataset: 1 problem
    dataset = Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "Translate this Python code to optimized Cython:\n```python\ndef sum_array(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total\n```",
                    }
                ]
            ],
        }
    )

    config = GRPOConfig(
        output_dir="/tmp/grpo_memory_test",
        num_generations=2,
        max_completion_length=512,
        max_tool_calling_iterations=2 if not no_tools else None,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-7,
        logging_steps=1,
        max_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        loss_type="cispo",
        scale_rewards="batch",
        use_vllm=use_vllm,
        vllm_mode="colocate" if use_vllm else None,
        vllm_gpu_memory_utilization=0.40 if use_vllm else None,
        report_to="none",
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # Simple reward function
    def simple_reward(completions, **kwargs):
        return [1.0 if len(c) > 10 else 0.0 for c in completions]

    trainer_kwargs = {
        "model": model_path,
        "args": config,
        "train_dataset": dataset,
        "reward_funcs": simple_reward,
        "peft_config": lora_config,
    }

    if not no_tools:
        # Use our CythonToolEnvironment
        from cnake_charmer.training.environment import CythonToolEnvironment

        trainer_kwargs["environment_factory"] = CythonToolEnvironment

    logger.info("Creating GRPOTrainer...")
    trainer = GRPOTrainer(**trainer_kwargs)
    print_gpu_memory("after trainer creation")

    logger.info("Running 1 training step...")
    try:
        trainer.train()
        logger.info("Training step completed successfully!")
        print_gpu_memory("after training step")
    except torch.cuda.OutOfMemoryError:
        logger.error("OUT OF MEMORY during training step!")
        print_gpu_memory("OOM state")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print_gpu_memory("error state")
        raise


def main():
    parser = argparse.ArgumentParser(description="GRPO memory smoke test")
    parser.add_argument(
        "--model",
        default="models/gpt-oss-20b-cython-sft-merged",
        help="Model path",
    )
    parser.add_argument("--use-vllm", action="store_true", help="Test with vLLM colocate")
    parser.add_argument("--no-tools", action="store_true", help="Skip tool execution")
    parser.add_argument(
        "--test",
        choices=["load", "generate", "trainer", "all"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    logger.info(f"Model: {args.model}")
    logger.info(f"vLLM: {args.use_vllm}")
    logger.info(f"Tools: {not args.no_tools}")
    print_gpu_memory("start")

    if args.test in ("load", "generate", "all"):
        model, tokenizer = test_model_loading(args.model)
        if args.test in ("generate", "all"):
            test_generation(model, tokenizer)
        # Free model memory before trainer test (or before exit for one-shot tests)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_memory("after cleanup")

    if args.test in ("trainer", "all"):
        test_grpo_trainer(args.model, use_vllm=args.use_vllm, no_tools=args.no_tools)

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
