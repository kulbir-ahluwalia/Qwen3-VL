#!/usr/bin/env python3
"""
finetune_qwen.py - Unsloth Fine-tuning Script for Qwen3-VL Waypoint Generation

Fine-tunes Qwen3-VL (8B or 32B) on waypoint generation data using:
- 4-bit quantization (bnb-4bit) for memory efficiency
- LoRA adapters for parameter-efficient training
- Multi-image support (RGB + BEV views)
- GGUF export for edge deployment (llama.cpp/Ollama)

Usage:
    # Single GPU
    python finetune_qwen.py \
        --model_size 8B \
        --dataset_path ./processed_data/train.pkl \
        --output_dir ./checkpoints

    # Multi-GPU with accelerate
    accelerate launch --num_processes=4 finetune_qwen.py \
        --model_size 8B \
        --dataset_path ./processed_data/train.pkl

    # With GGUF export
    python finetune_qwen.py \
        --model_size 8B \
        --dataset_path ./processed_data/train.pkl \
        --export_gguf
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL for waypoint generation using Unsloth"
    )

    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["8B", "32B"],
        default="8B",
        help="Qwen3-VL model size (default: 8B)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )

    # Data configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to processed train.pkl file"
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default=None,
        help="Path to processed val.pkl file (optional)"
    )

    # Training configuration (OPTIMIZED FOR DISTILLATION)
    # Based on KD-LoRA, Warmup-Distill, and Qwen VL research (2024-2025)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,  # Increased from 2 for stable coordinate regression
        help="Per-device batch size (default: 4, research: 12-16 optimal)"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=8,  # Increased from 4, effective batch = 32
        help="Gradient accumulation steps (default: 8, effective batch=32)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,  # Reduced from 5e-4 for distillation
        help="Learning rate (default: 2e-5, distillation optimal: 2e-6 to 2e-5)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,  # Increased from 3 for low-LR convergence
        help="Number of training epochs (default: 5, distillation needs more epochs)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for full epochs)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,  # Increased from 50 to bridge teacher-student gap
        help="Warmup steps (default: 500, Warmup-Distill research)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,  # Reduced from 0.01 to preserve teacher knowledge
        help="Weight decay (default: 0.001, lower for distillation)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,  # Gradient clipping for stability
        help="Max gradient norm for clipping (default: 0.5)"
    )

    # LoRA configuration (OPTIMIZED FOR DISTILLATION)
    # Based on KD-LoRA research: r=32 optimal for teacher-student distillation
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,  # Reduced from 64 (KD-LoRA: 99% param reduction, same perf)
        help="LoRA rank (default: 32, KD-LoRA research optimal for distillation)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,  # Alpha = 2×r heuristic
        help="LoRA alpha (default: 64, heuristic: 2×rank)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,  # Unsloth recommends 0.0 for vision models
        help="LoRA dropout (default: 0.0, Unsloth vision recommendation)"
    )

    # Fine-tuning scope
    parser.add_argument(
        "--finetune_vision",
        action="store_true",
        default=True,
        help="Fine-tune vision encoder (default: True)"
    )
    parser.add_argument(
        "--finetune_language",
        action="store_true",
        default=True,
        help="Fine-tune language model (default: True)"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--export_gguf",
        action="store_true",
        help="Export model to GGUF format after training"
    )
    parser.add_argument(
        "--gguf_quantization",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0"],
        help="GGUF quantization method (default: q4_k_m)"
    )

    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen3vl-waypoints",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging frequency (default: 10)"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides command-line args)"
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


# Model registry
MODEL_MAP = {
    "8B": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    "32B": "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit",
}


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load processed dataset from pickle file."""
    print(f"Loading dataset from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")
    return data


def setup_model(args: argparse.Namespace):
    """Load and configure the model with LoRA."""
    from unsloth import FastVisionModel

    model_name = MODEL_MAP[args.model_size]
    print(f"Loading model: {model_name}")

    # Load pre-quantized 4-bit model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_length,
    )

    # Apply LoRA
    print("Applying LoRA configuration...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision,
        finetune_language_layers=args.finetune_language,
        finetune_attention_modules=True,  # q, k, v, o
        finetune_mlp_modules=True,        # gate, up, down
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        use_rslora=False,
        loftq_config=None,
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def setup_trainer(
    model,
    tokenizer,
    train_dataset: List[Dict[str, Any]],
    val_dataset: Optional[List[Dict[str, Any]]],
    args: argparse.Namespace
):
    """Configure the SFT trainer."""
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    # Determine report_to
    report_to = ["wandb"] if os.environ.get("WANDB_API_KEY") else []

    # Training arguments (OPTIMIZED FOR DISTILLATION from Qwen-235B)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        warmup_steps=args.warmup_steps,
        warmup_ratio=0.1,  # 10% of training as warmup
        logging_steps=args.logging_steps,
        save_steps=200,  # Save less frequently
        save_total_limit=3,
        eval_steps=200,  # Evaluation frequency
        optim="paged_adamw_8bit",
        weight_decay=args.weight_decay,  # Now configurable (default: 0.001)
        max_grad_norm=args.max_grad_norm,  # Gradient clipping (default: 0.5)
        lr_scheduler_type="cosine_with_restarts",  # Better than plain cosine for distillation
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=report_to,
        run_name=args.wandb_run_name or f"qwen3vl-{args.model_size.lower()}-waypoints-distill",
        # Critical for vision models
        remove_unused_columns=False,
        skip_prepare_dataset=True,
        dataset_text_field="",  # Not used for vision
        max_seq_length=args.max_seq_length,
        # Gradient checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Data collator for multi-image
    data_collator = UnslothVisionDataCollator(model, tokenizer)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    return trainer


def export_gguf(model, tokenizer, args: argparse.Namespace):
    """Export model to GGUF format."""
    output_name = f"qwen3-vl-{args.model_size.lower()}-waypoints-{args.gguf_quantization}"
    output_path = Path(args.output_dir) / output_name

    print(f"Exporting to GGUF: {output_path}")
    print(f"Quantization: {args.gguf_quantization}")

    try:
        model.save_pretrained_gguf(
            str(output_path),
            tokenizer,
            quantization_method=args.gguf_quantization,
        )
        print(f"GGUF export complete: {output_path}")
    except Exception as e:
        print(f"Warning: GGUF export failed: {e}")
        print("Note: Vision model GGUF support may be limited")


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Print configuration
    print("=" * 60)
    print("Qwen3-VL Waypoint Fine-tuning with Unsloth")
    print("=" * 60)
    print(f"Model: {args.model_size} ({MODEL_MAP[args.model_size]})")
    print(f"Dataset: {args.dataset_path}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load datasets
    train_dataset = load_dataset(args.dataset_path)
    val_dataset = None
    if args.val_dataset_path:
        val_dataset = load_dataset(args.val_dataset_path)

    # Setup model
    model, tokenizer = setup_model(args)

    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset, args)

    # Train
    print("\nStarting training...")
    trainer_stats = trainer.train()

    # Print stats
    print("\nTraining complete!")
    print(f"Final loss: {trainer_stats.training_loss:.4f}")

    # Save final model
    print(f"\nSaving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Export to GGUF if requested
    if args.export_gguf:
        export_gguf(model, tokenizer, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
