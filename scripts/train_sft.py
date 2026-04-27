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


def _save_sharded(tensors: dict, output_dir: Path, max_shard_bytes: int = 5 * 1024**3):
    """Save a dict of tensors as sharded safetensors with index."""
    from safetensors.torch import save_file

    shards = []
    current_shard = {}
    current_size = 0
    for name, tensor in tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor.contiguous().cpu()
        current_size += tensor_size
    if current_shard:
        shards.append(current_shard)

    weight_map = {}
    total_size = 0
    for i, shard in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(output_dir / shard_name))
        for name, tensor in shard.items():
            weight_map[name] = shard_name
            total_size += tensor.numel() * tensor.element_size()
        logger.info(f"Saved {shard_name} ({len(shard)} tensors)")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def _copy_base_files(base_model_id: str, output_dir: Path, keep_quant_config: bool):
    """Copy config.json, tokenizer, chat template from base model cache.

    Don't use model.config.to_dict() — it renames rope_scaling to rope_parameters
    and nests rope_theta inside it, causing vLLM warnings.
    """
    base_cache = Path(
        snapshot_download(
            base_model_id,
            allow_patterns=["config.json", "generation_config*", "tokenizer*", "chat_template*"],
        )
    )
    base_config = json.loads((base_cache / "config.json").read_text())
    if not keep_quant_config:
        base_config.pop("quantization_config", None)
    base_config.pop("transformers_version", None)
    (output_dir / "config.json").write_text(json.dumps(base_config, indent=2))
    for pattern in ["generation_config*", "tokenizer*", "chat_template*"]:
        for f in base_cache.glob(pattern):
            shutil.copy2(f, output_dir / f.name)
    if SYSTEM_PROMPT_FILE.exists():
        shutil.copy2(SYSTEM_PROMPT_FILE, output_dir / "system_prompt.txt")


def _quantize_to_mxfp4(bf16_dir: Path, output_dir: Path, base_model_id: str):
    """Re-quantize a bf16 merged model to MXFP4, matching the base model format.

    Uses OpenAI's triton kernels (downcast_to_mxfp_torch) to quantize expert
    layers. Non-expert weights (attention, norms, embeddings) are copied as-is.
    The serialized format matches openai/gpt-oss-20b exactly so vLLM can load
    it with the Marlin MXFP4 backend.
    """
    from collections import defaultdict

    from safetensors import safe_open
    from transformers.integrations.hub_kernels import get_kernel

    tkh = get_kernel("kernels-community/gpt-oss-triton-kernels")
    downcast = tkh.numerics_details.mxfp.downcast_to_mxfp_torch

    output_dir.mkdir(parents=True, exist_ok=True)

    bf16_index = json.loads((bf16_dir / "model.safetensors.index.json").read_text())
    shard_keys = defaultdict(list)
    for key, shard in bf16_index["weight_map"].items():
        shard_keys[shard].append(key)

    # Collect all output tensors, then save sharded
    all_tensors = {}
    for shard_file in sorted(shard_keys.keys()):
        logger.info(f"Quantizing {shard_file}...")
        f = safe_open(str(bf16_dir / shard_file), framework="pt")

        for key in sorted(shard_keys[shard_file]):
            # Expert proj weights → quantize to MXFP4 blocks + scales
            if ".mlp.experts." in key and key.endswith(("gate_up_proj", "down_proj")):
                layer_name = key.rsplit(".", 1)[0]
                proj = key.rsplit(".", 1)[1]

                w = f.get_tensor(key).cuda()  # [experts, in, out]
                w_t = w.transpose(-1, -2).contiguous()  # [experts, out, in]
                blocks, scales = downcast(w_t.to(torch.bfloat16), torch.uint8, axis=2)
                blocks = blocks.reshape(blocks.shape[0], blocks.shape[1], -1, 16).cpu()
                scales = scales.cpu()

                all_tensors[f"{layer_name}.{proj}_blocks"] = blocks
                all_tensors[f"{layer_name}.{proj}_scales"] = scales

                del w, w_t, blocks, scales
                torch.cuda.empty_cache()
            else:
                # Non-expert weights pass through unchanged
                all_tensors[key] = f.get_tensor(key)

    _save_sharded(all_tensors, output_dir)
    _copy_base_files(base_model_id, output_dir, keep_quant_config=True)
    logger.info(f"MXFP4 model saved to {output_dir}")


def merge_and_save(base_model_id: str, adapter_path: str, output_dir: str):
    """Merge LoRA adapter into base model and save as both bf16 and MXFP4.

    Pipeline:
      1. Load base model dequantized to bf16
      2. Load and merge LoRA adapter
      3. Save bf16 merged model (for GRPO training)
      4. Re-quantize expert layers to MXFP4 (for serving and distribution)

    The bf16 save uses manual state_dict serialization because save_pretrained
    fails on dequantized MXFP4 models (NotImplementedError in revert_weight_conversion).
    """
    bf16_dir = Path(output_dir)
    mxfp4_dir = Path(str(output_dir).replace("-merged", "-mxfp4"))
    bf16_dir.mkdir(parents=True, exist_ok=True)

    # Step 1-2: Load dequantized base + merge LoRA
    logger.info(f"Loading base model {base_model_id} (dequantized to bf16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=True,
        device_map="auto",
    )

    logger.info(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    # Step 3: Save bf16
    logger.info(f"Saving bf16 merged model to {bf16_dir}...")
    _save_sharded(model.state_dict(), bf16_dir)
    _copy_base_files(base_model_id, bf16_dir, keep_quant_config=False)
    logger.info(f"bf16 model saved to {bf16_dir}")

    # Free the model to reclaim GPU memory for quantization
    del model, base_model
    torch.cuda.empty_cache()

    # Step 4: Re-quantize to MXFP4
    logger.info("Re-quantizing to MXFP4...")
    _quantize_to_mxfp4(bf16_dir, mxfp4_dir, base_model_id)
    logger.info(f"Done! bf16: {bf16_dir}, MXFP4: {mxfp4_dir}")


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

    # LoRA on all linear layers + expert MoE layers at every 4th layer.
    # target_parameters adds LoRA to expert gate_up_proj/down_proj on dequantized model.
    # (Warning "no parameter was matched" appears only during merge with non-dequantized model;
    #  during training with dequantize=True, experts are standard modules and it works.)
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
        logging_dir=str(Path(args.output) / "runs"),
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
        save_steps=25,
        save_total_limit=3,
        report_to=["tensorboard"],
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

    # Merge the final checkpoint (highest step number)
    if not args.no_merge:
        output_dir = Path(args.output)
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
        )
        adapter_path = str(checkpoints[-1]) if checkpoints else args.output

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
