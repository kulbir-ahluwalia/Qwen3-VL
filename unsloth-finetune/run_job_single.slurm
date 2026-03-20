#!/bin/bash
#SBATCH --job-name=qwen3vl-test
#SBATCH --partition=ghx4
#SBATCH --account=[DELTA_ACCOUNT]
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================================================
# Qwen3-VL Waypoint Fine-tuning - Single GPU Test Job
# ============================================================================
# Use this for quick testing before running full training
# Target: 1x H100 80GB GPU
# ============================================================================

echo "============================================"
echo "Single GPU Test Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

mkdir -p logs

# Environment setup
module purge
module load python/miniforge3_pytorch
module load cuda/12.2

if [ -d "unsloth_env" ]; then
    source unsloth_env/bin/activate
else
    echo "Error: unsloth_env not found. Run setup first."
    exit 1
fi

# HuggingFace settings
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# Quick Test Run (5 steps)
# ============================================================================

MODEL_SIZE=${1:-"8B"}
DATASET_PATH=${2:-"./processed_data/train.pkl"}
OUTPUT_DIR="./test_checkpoints/$SLURM_JOB_ID"

mkdir -p $OUTPUT_DIR

python finetune_qwen.py \
    --model_size $MODEL_SIZE \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --gradient_accumulation 2 \
    --max_steps 5 \
    --learning_rate 5e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --warmup_steps 1 \
    --logging_steps 1

echo "============================================"
echo "Test complete!"
echo "End time: $(date)"
echo "============================================"
