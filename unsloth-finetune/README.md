# Unsloth Qwen3-VL Fine-tuning for Waypoint Generation

Fine-tune Qwen3-VL (8B/32B) on natural language waypoint navigation commands using Unsloth for 2x faster training with 60% less VRAM.

## Overview

This pipeline trains a vision-language model to:
1. **Input**: RGB image + Bird's-eye view (BEV) + natural language command
2. **Output**: 2D waypoints + target bounding box

## Quick Start

### 1. Prepare Data

Convert Schema 1.0 JSON to Unsloth format:

```bash
python prepare_unsloth_data.py \
    --dataset_json /path/to/waypoint_dataset.json \
    --output_dir ./processed_data \
    --train_split 0.9 \
    --validate_paths
```

### 2. Train (Local)

Single GPU:
```bash
python finetune_qwen.py \
    --model_size 8B \
    --dataset_path ./processed_data/train.pkl \
    --val_dataset_path ./processed_data/val.pkl \
    --output_dir ./checkpoints \
    --export_gguf
```

### 3. Train (Delta-AI)

Submit to Slurm:
```bash
# Edit run_job.slurm: Replace [DELTA_ACCOUNT] and [YOUR_EMAIL]
sbatch run_job.slurm 8B ./processed_data/train.pkl ./processed_data/val.pkl
```

## Files

| File | Description |
|------|-------------|
| `prepare_unsloth_data.py` | Convert Schema 1.0 JSON to Unsloth pickle |
| `finetune_qwen.py` | Main training script |
| `run_job.slurm` | Delta-AI 4x H100 job |
| `run_job_single.slurm` | Single GPU test job |
| `configs/config_8b.yaml` | 8B model hyperparameters |
| `configs/config_32b.yaml` | 32B model hyperparameters |
| `configs/ds_zero3.json` | DeepSpeed ZeRO-3 config |

## Schema 1.0 Format

Input JSON structure:
```json
{
  "schema_version": "1.0",
  "images": [
    {
      "rgb_path": "data/images/rgb/000001.png",
      "bev_path": "data/images/bev/000001.png",
      "prompts": [
        {
          "nl_command": "Go to the red bucket near the table.",
          "parsed_output_text": "[(0.0,0.0,0.0),(2.0,0.0,0.0)]",
          "target": {
            "bbox": { "values": [412, 180, 512, 300] }
          }
        }
      ]
    }
  ]
}
```

## Model Configurations

| Model | VRAM (4-bit) | Batch Size | GPUs |
|-------|--------------|------------|------|
| Qwen3-VL-8B | ~6.5GB | 2 | 1x H100 |
| Qwen3-VL-32B | ~18GB | 1 | 4x H100 |

## Using Config Files

```bash
python finetune_qwen.py --config configs/config_8b.yaml
```

## GGUF Export

After training, the model is exported to GGUF format for use with:
- **llama.cpp**: `./llama-cli -m qwen3-vl-8b-waypoints-q4km.gguf`
- **Ollama**: `ollama create waypoint-model -f Modelfile`

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` to 1
- Increase `--gradient_accumulation`
- Use DeepSpeed ZeRO-3 for 32B model

### Multi-Image Training Error
The prepare script uses list comprehension instead of `.map()` to avoid:
```
AttributeError: 'dict' has no attribute 'startswith'
```

### Slow Training
- Ensure Flash Attention 2 is installed: `pip install flash-attn --no-build-isolation`
- Verify bf16 is enabled (check GPU support)

## References

- [Unsloth Vision Fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning)
- [Qwen3-VL Documentation](https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune)
- [Delta-AI User Guide](https://docs.ncsa.illinois.edu/systems/delta/)
