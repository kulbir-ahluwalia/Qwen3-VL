# CLAUDE.md - Qwen3-VL

Alibaba's advanced vision-language model combining text understanding, visual perception, video reasoning, spatial grounding, and agent interaction. Available in Dense and MoE architectures from 2B to 235B parameters.

## Project Structure

```
Qwen3-VL/
├── README.md                     # Main documentation
├── web_demo_mm.py                # Gradio web UI demo
├── requirements_web_demo.txt     # Web demo dependencies
├── qwen-vl-finetune/             # Fine-tuning framework
│   ├── qwenvl/
│   │   ├── train/                # Training scripts
│   │   └── data/                 # Data processing
│   ├── scripts/                  # Training shell scripts
│   └── demo/                     # Sample data
├── qwen-vl-utils/                # Standalone utility library
├── cookbooks/                    # 13 interactive notebooks
├── evaluation/                   # Benchmark reproduction code
└── docker/                       # Container support
```

## Supported Models

| Model | Size | Type | Architecture |
|-------|------|------|--------------|
| Qwen3-VL-235B-A22B | 235B | Instruct/Thinking | MoE |
| Qwen3-VL-32B | 32B | Instruct/Thinking | Dense |
| Qwen3-VL-8B | 8B | Instruct/Thinking | Dense |
| Qwen3-VL-4B | 4B | Instruct/Thinking | Dense |
| Qwen3-VL-2B | 2B | Instruct/Thinking | Dense |
| Qwen3-VL-30B-A3B | 30B | Instruct/Thinking | MoE |

All models support 256K context (expandable to 1M with YaRN).

## Installation

```bash
# Core dependencies
pip install "transformers>=4.57.0"
pip install torch torchvision accelerate
pip install qwen-vl-utils==0.0.14

# Optional: Flash Attention for speedup
pip install -U flash-attn --no-build-isolation

# Optional: vLLM for inference
pip install "vllm>=0.11.0"
```

## Quick Start

### Web Demo
```bash
pip install -r requirements_web_demo.txt
python web_demo_mm.py -c Qwen/Qwen3-VL-4B-Instruct --server-port 7860
```

### Inference with Transformers
```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "file:///path/to/image.jpg"},
        {"type": "text", "text": "Describe this image."}
    ]
}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(output, skip_special_tokens=True))
```

### vLLM Deployment
```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 --port 22002
```

## Fine-tuning

### Setup
```bash
cd qwen-vl-finetune
pip install torch==2.6.0 torchvision==0.21.0
pip install transformers==4.57.0.dev0 deepspeed==0.17.1
pip install flash_attn==2.7.4.post1 accelerate==1.7.0 peft==0.17.1
```

### Data Format

**Single Image:**
```json
{
    "image": "images/001.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat's this?"},
        {"from": "gpt", "value": "An apple on a table"}
    ]
}
```

**Video:**
```json
{
    "video": "videos/005.mp4",
    "conversations": [
        {"from": "human", "value": "<video>\nWhat happens?"},
        {"from": "gpt", "value": "..."}
    ]
}
```

**Grounding (2D bbox):**
```json
{
    "image": "demo/image.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nLocate house"},
        {"from": "gpt", "value": "{\"bbox_2d\": [135, 114, 1016, 672]}"}
    ]
}
```

### Training Command
```bash
bash scripts/sft_qwen3_4b.sh
```

### Key Training Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `tune_mm_vision` | False | Train vision encoder |
| `tune_mm_mlp` | True | Train projection layers |
| `tune_mm_llm` | True | Train LLM backbone |
| `max_pixels` | 50176 | Max image resolution |
| `video_max_frames` | 8 | Max video frames |
| `lora_enable` | False | Use LoRA adapters |

## Cookbooks (13 Notebooks)

| Notebook | Use Case |
|----------|----------|
| `omni_recognition.ipynb` | Object/animal/plant recognition |
| `document_parsing.ipynb` | Document layout extraction |
| `2d_grounding.ipynb` | Bounding box localization |
| `ocr.ipynb` | 32-language OCR |
| `video_understanding.ipynb` | Video analysis |
| `mobile_agent.ipynb` | Mobile UI automation |
| `computer_use.ipynb` | Desktop automation |
| `3d_grounding.ipynb` | 3D bounding boxes |
| `think_with_images.ipynb` | Visual reasoning |
| `mmcode.ipynb` | Code from visuals |
| `spatial_understanding.ipynb` | Spatial reasoning |

## Key Features

- **Interleaved-MRoPE**: Full-frequency position encoding for video
- **DeepStack**: Multi-level ViT features for fine-grained details
- **Text-Timestamp Alignment**: Precise event localization
- **256K Context**: Native support, expandable to 1M

## Video Processing

```python
# Frame sampling options
messages = [{
    "role": "user",
    "content": [{
        "type": "video",
        "video": "file:///path/to/video.mp4",
        "fps": 2.0,  # or num_frames=N
    }]
}]
```

Backends: torchcodec (recommended), torchvision, decord

## Flash Attention 2

```python
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
```

## Docker

```bash
docker run --gpus all --ipc=host --network=host --rm \
    --name qwen3vl -it qwenllm/qwenvl:qwen3vl-cu128 bash
```

## API Access (DashScope)

```python
from openai import OpenAI

client = OpenAI(
    api_key="your_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
response = client.chat.completions.create(
    model="qwen3-vl-235b-a22b-instruct",
    messages=[...]
)
```

## Evaluation

Benchmark reproduction code in `evaluation/`:
- MMMU - Multimodal multi-task
- MathVision - Mathematical reasoning
- ODinW-13 - Object detection
- RealWorldQA - Real-world QA

## License

Apache License 2.0
