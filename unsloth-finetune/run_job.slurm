#!/bin/bash
#SBATCH --job-name=qwen3vl-waypoints
#SBATCH --partition=ghx4
#SBATCH --account=[DELTA_ACCOUNT]
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=[YOUR_EMAIL]

# ============================================================================
# Qwen3-VL Waypoint Fine-tuning - Delta-AI Multi-GPU Job
# ============================================================================
# Target: 4x H100 80GB GPUs on ghx4 partition
# Framework: Unsloth + accelerate for distributed training
# ============================================================================

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "============================================"

# Create logs directory
mkdir -p logs

# ============================================================================
# Environment Setup
# ============================================================================

# Load modules (Delta-AI specific)
module purge
module load python/miniforge3_pytorch
module load cuda/12.2

# Activate virtual environment (create if needed)
if [ ! -d "unsloth_env" ]; then
    echo "Creating virtual environment..."
    python -m venv unsloth_env
    source unsloth_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source unsloth_env/bin/activate
fi

# ============================================================================
# NCCL Configuration for Slingshot11 Interconnect
# ============================================================================

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3

# HuggingFace settings
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface
export HF_HOME=/scratch/$USER/.cache/huggingface

# Disable tokenizers parallelism (avoids deadlocks)
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# Training Configuration
# ============================================================================

MODEL_SIZE=${1:-"8B"}
DATASET_PATH=${2:-"./processed_data/train.pkl"}
VAL_DATASET_PATH=${3:-"./processed_data/val.pkl"}
OUTPUT_DIR=${4:-"./checkpoints/$SLURM_JOB_ID"}

echo "Model size: $MODEL_SIZE"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# ============================================================================
# Launch Training with Accelerate (4x GPU)
# ============================================================================

accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    finetune_qwen.py \
    --model_size $MODEL_SIZE \
    --dataset_path $DATASET_PATH \
    --val_dataset_path $VAL_DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 2 \
    --gradient_accumulation 4 \
    --num_epochs 3 \
    --learning_rate 5e-4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --warmup_steps 50 \
    --logging_steps 10 \
    --export_gguf \
    --wandb_project "qwen3vl-waypoints" \
    --wandb_run_name "delta-${MODEL_SIZE}-${SLURM_JOB_ID}"

# ============================================================================
# Post-training
# ============================================================================

echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================"

# Copy logs to output directory
cp logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out $OUTPUT_DIR/ 2>/dev/null || true
cp logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err $OUTPUT_DIR/ 2>/dev/null || true

echo "Done!"
