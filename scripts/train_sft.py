"""
SFT training for gpt-oss-20b on Python→Cython conversion.

Fine-tunes the model using LoRA on expert layers with multi-turn tool-use traces.
Based on: https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers

Usage:
    # Dry run (5 steps)
    uv run --no-sync python scripts/train_sft.py --max-steps 5

    # Full training
    uv run --no-sync python scripts/train_sft.py

    # With OOM fallbacks
    uv run --no-sync python scripts/train_sft.py --max-seq-length 8192 --lora-rank 32
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_DATASET = "data/sft_dataset.jsonl"
DEFAULT_OUTPUT = "models/gpt-oss-20b-cython-sft"
SYSTEM_PROMPT_FILE = Path("data/system_prompt.txt")


def load_dataset_jsonl(path: str) -> Dataset:
    """Load SFT dataset from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                examples.append({"messages": ex["messages"]})
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(description="SFT training for gpt-oss-20b")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help=f"Dataset path (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help=f"Output dir (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--max-length", type=int, default=16384, help="Max sequence length (default: 16384)"
    )
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (default: 64)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Per-device batch size (default: 1)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max steps (-1 for full run, use 5 for dry run)"
    )
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing")
    parser.add_argument(
        "--no-merge", action="store_true", help="Skip merge step (save adapter only)"
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset_jsonl(args.dataset)

    # Dequantize MXFP4 → bf16 for training (MXFP4 doesn't support training)
    # Use flex_attention for O(N) memory with sink attention (per HF docs recommendation)
    logger.info(f"Loading model: {args.model}")
    quantization_config = Mxfp4Config(dequantize=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="flex_attention",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            use_cache=False,
            device_map="auto",
        )
        logger.info("Loaded with flex_attention")
    except ValueError:
        logger.info("flex_attention init failed, loading eager and overriding")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            use_cache=False,
            device_map="auto",
        )
        model.config._attn_implementation = "flex_attention"

    # Apply liger kernels for memory-efficient non-attention ops
    from liger_kernel.transformers import apply_liger_kernel_to_gpt_oss

    apply_liger_kernel_to_gpt_oss(model, fused_linear_cross_entropy=False)
    logger.info("Applied liger kernels")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config targeting expert layers (per OpenAI cookbook)
    lora_alpha = args.lora_rank * 2
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training config
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        warmup_ratio=0.1,
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="none",
        max_steps=args.max_steps,
        packing=args.packing,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_merge:
        logger.info(f"Saving adapter to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        logger.info("Merging LoRA adapter into base model...")
        merged = model.merge_and_unload()
        logger.info(f"Saving merged model to {output_dir}")
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Copy system prompt alongside model
    if SYSTEM_PROMPT_FILE.exists():
        shutil.copy2(SYSTEM_PROMPT_FILE, output_dir / "system_prompt.txt")
        logger.info("Copied system_prompt.txt to output dir")

    logger.info("Done!")


if __name__ == "__main__":
    main()
