#!/usr/bin/env python3
"""
prepare_unsloth_data.py - Convert Schema 1.0 Waypoint JSON to Unsloth VLM Format

Transforms the waypoint generation dataset (Qwen-235B + SAM2 output) into the
multi-image format required by Unsloth's FastVisionModel for Qwen3-VL fine-tuning.

Schema 1.0 Input:
    - images[].rgb_path: First-person RGB view (semantics)
    - images[].bev_path: Bird's-eye view (geometry)
    - images[].prompts[].nl_command: Natural language navigation command
    - images[].prompts[].parsed_output_text: Waypoints as coordinate list
    - images[].prompts[].target.bbox.values: Target bounding box [x1, y1, x2, y2]

Unsloth Output Format:
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": <PIL.Image>},  # RGB
                {"type": "image", "image": <PIL.Image>},  # BEV
                {"type": "text", "text": "Command: {nl_command}"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Waypoints: {waypoints} Bbox: {bbox}"}
            ]}
        ]
    }

Usage:
    python prepare_unsloth_data.py \
        --dataset_json /path/to/waypoint_dataset.json \
        --output_dir ./processed_data \
        --train_split 0.9 \
        --seed 42 \
        --validate_paths
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Schema 1.0 Waypoint JSON to Unsloth VLM format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to the Schema 1.0 waypoint dataset JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Directory to save processed train/val pickle files (default: ./processed_data)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--validate_paths",
        action="store_true",
        help="Validate that rgb_path and bev_path exist before processing"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory for relative image paths (default: directory containing JSON)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        help="Resize images to (width, height). Default: keep original size"
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip samples with missing images instead of raising error"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate dataset without saving (for testing)"
    )
    return parser.parse_args()


def load_schema_1_json(json_path: str) -> Dict[str, Any]:
    """Load and validate Schema 1.0 JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Validate schema version
    schema_version = data.get("schema_version", "unknown")
    if schema_version != "1.0":
        print(f"Warning: Expected schema_version 1.0, got {schema_version}")

    # Validate required fields
    if "images" not in data:
        raise ValueError("JSON must contain 'images' array")

    return data


def resolve_image_path(path: str, data_root: Path) -> Path:
    """Resolve image path relative to data root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return data_root / path


def load_image(path: Path, resize: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Load image as PIL.Image in RGB mode."""
    img = Image.open(path).convert("RGB")
    if resize:
        img = img.resize(resize, Image.Resampling.LANCZOS)
    return img


def convert_sample_to_unsloth(
    image_entry: Dict[str, Any],
    prompt_entry: Dict[str, Any],
    data_root: Path,
    validate_paths: bool = False,
    skip_missing: bool = False,
    resize: Optional[Tuple[int, int]] = None
) -> Optional[Dict[str, Any]]:
    """
    Convert a single image+prompt pair to Unsloth format.

    Returns None if images are missing and skip_missing=True.
    """
    # Resolve image paths
    rgb_path = resolve_image_path(image_entry["rgb_path"], data_root)
    bev_path = resolve_image_path(image_entry["bev_path"], data_root)

    # Validate paths exist
    if validate_paths or not skip_missing:
        for path, name in [(rgb_path, "rgb_path"), (bev_path, "bev_path")]:
            if not path.exists():
                if skip_missing:
                    print(f"Skipping: {name} not found: {path}")
                    return None
                raise FileNotFoundError(f"{name} not found: {path}")

    # Load images
    try:
        rgb_image = load_image(rgb_path, resize)
        bev_image = load_image(bev_path, resize)
    except Exception as e:
        if skip_missing:
            print(f"Skipping: Error loading images: {e}")
            return None
        raise

    # Extract ground truth
    nl_command = prompt_entry["nl_command"]
    waypoints = prompt_entry.get("parsed_output_text", "")

    # Extract bounding box
    bbox = prompt_entry.get("target", {}).get("bbox", {}).get("values", [])
    bbox_str = str(bbox) if bbox else "[]"

    # Construct Unsloth message format
    # CRITICAL: Use list of dicts with PIL.Image objects for multi-image
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "image", "image": bev_image},
                    {"type": "text", "text": f"Command: {nl_command}"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Waypoints: {waypoints} Bbox: {bbox_str}"}
                ]
            }
        ]
    }

    return sample


def process_dataset(
    data: Dict[str, Any],
    data_root: Path,
    args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """
    Process all images and prompts into Unsloth format.

    IMPORTANT: Uses list comprehension instead of .map() to avoid
    HuggingFace Arrow serialization errors with PIL.Image objects.
    """
    samples = []
    skipped = 0

    images = data["images"]
    if args.max_samples:
        images = images[:args.max_samples]

    print(f"Processing {len(images)} images...")

    for image_entry in tqdm(images, desc="Converting"):
        for prompt_entry in image_entry.get("prompts", []):
            sample = convert_sample_to_unsloth(
                image_entry=image_entry,
                prompt_entry=prompt_entry,
                data_root=data_root,
                validate_paths=args.validate_paths,
                skip_missing=args.skip_missing,
                resize=tuple(args.image_size) if args.image_size else None
            )

            if sample is not None:
                samples.append(sample)
            else:
                skipped += 1

    print(f"Converted {len(samples)} samples, skipped {skipped}")
    return samples


def split_dataset(
    samples: List[Dict[str, Any]],
    train_split: float,
    seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split samples into train and validation sets."""
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)

    split_idx = int(len(samples) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    return train_samples, val_samples


def save_dataset(samples: List[Dict[str, Any]], output_path: Path) -> None:
    """Save samples to pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(samples)} samples to {output_path}")


def print_sample_info(samples: List[Dict[str, Any]], name: str) -> None:
    """Print information about a sample for verification."""
    print(f"\n{name} Dataset:")
    print(f"  Total samples: {len(samples)}")

    if samples:
        sample = samples[0]
        user_msg = sample["messages"][0]
        asst_msg = sample["messages"][1]

        print(f"  Sample structure:")
        print(f"    User content items: {len(user_msg['content'])}")
        for i, item in enumerate(user_msg['content']):
            if item['type'] == 'image':
                img = item['image']
                print(f"      [{i}] image: {img.size} {img.mode}")
            else:
                text = item['text'][:50] + "..." if len(item['text']) > 50 else item['text']
                print(f"      [{i}] text: {text}")

        asst_text = asst_msg['content'][0]['text']
        asst_preview = asst_text[:80] + "..." if len(asst_text) > 80 else asst_text
        print(f"    Assistant: {asst_preview}")


def main():
    args = parse_args()

    # Determine data root
    json_path = Path(args.dataset_json)
    if not json_path.exists():
        print(f"Error: Dataset JSON not found: {json_path}")
        sys.exit(1)

    data_root = Path(args.data_root) if args.data_root else json_path.parent

    print(f"Loading dataset from: {json_path}")
    print(f"Data root: {data_root}")

    # Load Schema 1.0 JSON
    data = load_schema_1_json(str(json_path))

    dataset_info = data.get("dataset", {})
    print(f"Dataset: {dataset_info.get('name', 'unknown')}")
    print(f"Created: {dataset_info.get('created_utc', 'unknown')}")
    print(f"Total images: {len(data['images'])}")

    total_prompts = sum(len(img.get('prompts', [])) for img in data['images'])
    print(f"Total prompts: {total_prompts}")

    # Process dataset
    # CRITICAL: This uses list comprehension internally, NOT .map()
    samples = process_dataset(data, data_root, args)

    if not samples:
        print("Error: No samples were successfully processed")
        sys.exit(1)

    # Split into train/val
    train_samples, val_samples = split_dataset(
        samples, args.train_split, args.seed
    )

    print(f"\nSplit: {len(train_samples)} train, {len(val_samples)} val")

    # Print sample info
    print_sample_info(train_samples, "Train")
    print_sample_info(val_samples, "Validation")

    if args.dry_run:
        print("\nDry run complete. No files saved.")
        return

    # Save datasets
    output_dir = Path(args.output_dir)
    save_dataset(train_samples, output_dir / "train.pkl")
    save_dataset(val_samples, output_dir / "val.pkl")

    # Save metadata
    metadata = {
        "source_json": str(json_path),
        "data_root": str(data_root),
        "schema_version": data.get("schema_version", "1.0"),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_split": args.train_split,
        "seed": args.seed,
        "image_size": args.image_size,
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {output_dir / 'metadata.json'}")

    print("\nDone! Files created:")
    print(f"  {output_dir / 'train.pkl'}")
    print(f"  {output_dir / 'val.pkl'}")
    print(f"  {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
