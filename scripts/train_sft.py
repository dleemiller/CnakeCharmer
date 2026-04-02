"""
SFT training for gpt-oss-20b on Python→Cython conversion.

Fine-tunes the model using LoRA on expert layers with multi-turn tool-use traces.
Based on: https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers
         https://github.com/huggingface/gpt-oss-recipes

Usage:
    # Dry run (5 steps)
    uv run --no-sync python scripts/train_sft.py --max-steps 5 --no-merge

    # Full training (1 epoch recommended — converges fast)
    uv run --no-sync python scripts/train_sft.py --epochs 1

    # Merge a specific checkpoint without training
    uv run --no-sync python scripts/train_sft.py --merge-only --checkpoint models/.../checkpoint-100
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_DATASET = "data/sft_dataset.jsonl"
DEFAULT_OUTPUT = "models/gpt-oss-20b-cython-sft"
SYSTEM_PROMPT_FILE = Path("data/system_prompt.txt")


def load_dataset_jsonl(path: str) -> Dataset:
    """Load SFT dataset from JSONL file.

    Expects pre-rendered Harmony text in the "text" column.
    """
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                examples.append({"text": ex["text"]})
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return Dataset.from_list(examples)


def merge_and_save(base_model_id: str, adapter_path: str, output_dir: str):
    """Merge LoRA adapter into base model and save.

    Loads the base model in native MXFP4 (not dequantized) so the merged
    model stays compact. Copies tokenizer directly from the base model
    cache to avoid format mismatches between transformers versions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model {base_model_id} (native MXFP4)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        attn_implementation="eager",
        torch_dtype="auto",
        use_cache=True,
        device_map="auto",
    )

    logger.info(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")

    # Copy tokenizer + chat template from base model cache (avoids format mismatch)
    base_cache = Path(
        snapshot_download(base_model_id, allow_patterns=["tokenizer*", "chat_template*"])
    )
    for pattern in ["tokenizer*", "chat_template*"]:
        for f in base_cache.glob(pattern):
            shutil.copy2(f, output_dir / f.name)
    logger.info("Copied tokenizer from base model cache")

    # Copy system prompt
    if SYSTEM_PROMPT_FILE.exists():
        shutil.copy2(SYSTEM_PROMPT_FILE, output_dir / "system_prompt.txt")

    logger.info(f"Done! Merged model saved to {output_dir}")


def train(args):
    """Run SFT training."""
    dataset = load_dataset_jsonl(args.dataset)

    # Load model dequantized for training, with flex_attention for O(N) memory
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

    # LoRA targeting attention (all layers) + experts (every 4th layer)
    lora_alpha = args.lora_rank * 2
    expert_targets = []
    for layer in [3, 7, 11, 15, 19, 23]:
        expert_targets.append(f"{layer}.mlp.experts.gate_up_proj")
        expert_targets.append(f"{layer}.mlp.experts.down_proj")
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        target_parameters=expert_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
        report_to=["trackio"],
        log_level="info",
        max_steps=args.max_steps,
        packing=args.packing,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    # Save adapter from best checkpoint
    if not args.no_merge:
        output_dir = Path(args.output)
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
        )
        adapter_path = str(checkpoints[0]) if checkpoints else args.output

        merge_dir = str(output_dir) + "-merged"
        merge_and_save(args.model, adapter_path, merge_dir)


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
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1)")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Per-device batch size (default: 1)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max steps (-1 for full, 5 for dry run)"
    )
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing")
    parser.add_argument(
        "--no-merge", action="store_true", help="Skip merge step (save adapter only)"
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge a checkpoint without training",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Adapter checkpoint path for --merge-only",
    )
    args = parser.parse_args()

    if args.merge_only:
        if not args.checkpoint:
            parser.error("--merge-only requires --checkpoint")
        merge_dir = str(Path(args.output)) + "-merged"
        merge_and_save(args.model, args.checkpoint, merge_dir)
    else:
        train(args)


if __name__ == "__main__":
    main()
