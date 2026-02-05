#!/usr/bin/env python
"""Run PatchCore inference and generate JSON for LLM evaluation.

This script processes all images in mmad_10classes.json and outputs
anomaly detection results in a format suitable for LLM evaluation.

Usage:
    # Run inference on all images
    python patchcore_training/scripts/inference.py

    # Run with custom config and output
    python patchcore_training/scripts/inference.py \
        --config patchcore_training/config/config.yaml \
        --output output/patchcore_predictions.json

    # Test with limited images
    python patchcore_training/scripts/inference.py --max-images 100

    # Resume from existing output
    python patchcore_training/scripts/inference.py --resume
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add patchcore_training to path
SCRIPT_DIR = Path(__file__).resolve().parent
PATCHCORE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PATCHCORE_ROOT))

from src.model import PatchCore
from src.utils import load_config, setup_seed, get_device, normalize_anomaly_map, compute_defect_location
from src.dataset import InferenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run PatchCore inference for LLM evaluation")

    parser.add_argument(
        "--config",
        type=str,
        default=str(PATCHCORE_ROOT / "config" / "config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (overrides config)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images to process (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def load_models(config: Dict, device: torch.device) -> Dict[str, PatchCore]:
    """Load all trained PatchCore models.

    Args:
        config: Configuration dictionary
        device: Device to load models to

    Returns:
        Dictionary mapping "dataset/category" to models
    """
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    datasets_config = config["data"].get("datasets", {})

    # Count total categories
    all_categories = [(d, c) for d, cats in datasets_config.items() for c in cats]

    models = {}

    for dataset_name, category in tqdm(all_categories, desc="Loading models"):
        pt_path = checkpoint_dir / dataset_name / category / "model.pt"

        if pt_path.exists():
            model = PatchCore.load(str(pt_path), device)
            model.eval()
            key = f"{dataset_name}/{category}"
            models[key] = model
        # Silently skip missing models (will be reported during inference)

    print(f"Loaded {len(models)}/{len(all_categories)} models")
    return models


def process_batch(
    model: PatchCore,
    images: torch.Tensor,
    original_sizes: List[tuple],
    threshold: float,
    device: torch.device,
) -> List[Dict]:
    """Process a batch of images.

    Args:
        model: PatchCore model
        images: Batch of images (B, C, H, W)
        original_sizes: List of original image sizes
        threshold: Anomaly threshold
        device: Device

    Returns:
        List of result dictionaries
    """
    images = images.to(device)

    with torch.no_grad():
        scores, maps = model.predict(images)

    scores = scores.cpu().numpy()
    maps = maps.cpu().numpy()

    results = []
    for i in range(len(scores)):
        # Normalize map
        anomaly_map = maps[i]
        anomaly_map_norm = normalize_anomaly_map(anomaly_map)

        # Resize map to original size
        orig_h, orig_w = original_sizes[i]
        if orig_h > 0 and orig_w > 0:
            anomaly_map_resized = cv2.resize(
                anomaly_map_norm,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            anomaly_map_resized = anomaly_map_norm

        # Normalize score to [0, 1]
        score = float(scores[i])

        # Compute defect location
        location_info = compute_defect_location(anomaly_map_norm, threshold)

        result = {
            "anomaly_score": round(score, 6),
            "is_anomaly": location_info["has_defect"],
            "threshold": threshold,
            "defect_location": location_info,
            "map_stats": {
                "max": round(float(anomaly_map_norm.max()), 4),
                "mean": round(float(anomaly_map_norm.mean()), 4),
                "std": round(float(anomaly_map_norm.std()), 4),
            },
        }

        results.append(result)

    return results


def main():
    args = parse_args()

    # Set seed
    setup_seed(args.seed)

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Get device
    device = get_device(config.get("device", "auto"))
    print(f"Using device: {device}")

    # Paths
    data_root = Path(config["data"]["root"])
    mmad_json_path = Path(config["data"]["mmad_json"])
    output_path = Path(args.output or config["output"]["inference_output"])
    threshold = config.get("evaluation", {}).get("threshold", 0.5)

    if not mmad_json_path.exists():
        print(f"Error: mmad.json not found: {mmad_json_path}")
        return

    # Load MMAD data
    print(f"Loading MMAD data from: {mmad_json_path}")
    with open(mmad_json_path, "r", encoding="utf-8") as f:
        mmad_data = json.load(f)

    image_paths = list(mmad_data.keys())
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Total images to process: {len(image_paths)}")

    # Load existing results if resuming
    existing_results = {}
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_list = json.load(f)
            existing_results = {r["image_path"]: r for r in existing_list}
        print(f"Loaded {len(existing_results)} existing results")

    # Filter out already processed
    remaining_paths = [p for p in image_paths if p not in existing_results]
    print(f"Remaining images: {len(remaining_paths)}")

    if not remaining_paths:
        print("All images already processed!")
        return

    # Load models
    print("\nLoading models...")
    models = load_models(config, device)

    if not models:
        print("No trained models found!")
        return

    # Group images by category
    images_by_category = {}
    for path in remaining_paths:
        parts = path.split("/")
        if len(parts) >= 2:
            key = f"{parts[0]}/{parts[1]}"
            if key not in images_by_category:
                images_by_category[key] = []
            images_by_category[key].append(path)

    # Process images
    results = list(existing_results.values())
    processed = len(existing_results)
    skipped = 0
    errors = 0

    print(f"\n{'='*60}")
    print("Running inference")
    print(f"{'='*60}")

    category_pbar = tqdm(images_by_category.items(), desc="Categories", position=0)
    for category_key, paths in category_pbar:
        if category_key not in models:
            skipped += len(paths)
            category_pbar.set_postfix({"status": "skipped (no model)"})
            continue

        model = models[category_key]
        dataset_name, category = category_key.split("/")

        category_pbar.set_description(f"Processing {category_key}")

        # Create dataset and dataloader
        dataset = InferenceDataset(
            data_root=data_root,
            image_paths=paths,
            image_size=config["data"].get("image_size", 224),
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        for batch in tqdm(dataloader, desc=f"  {category_key}", position=1, leave=False):
            valid_mask = batch["valid"]

            if not valid_mask.any():
                errors += (~valid_mask).sum().item()
                continue

            # Get valid images only
            valid_indices = torch.where(valid_mask)[0]
            images = batch["image"][valid_indices]
            original_sizes = [batch["original_size"][i] for i in valid_indices.tolist()]
            original_sizes = [(int(h), int(w)) for h, w in zip(
                [s[0] for s in original_sizes],
                [s[1] for s in original_sizes]
            )]

            # Process batch
            try:
                batch_results = process_batch(
                    model=model,
                    images=images,
                    original_sizes=original_sizes,
                    threshold=threshold,
                    device=device,
                )

                # Add metadata and append results
                for idx, result in zip(valid_indices.tolist(), batch_results):
                    result["image_path"] = batch["image_path"][idx]
                    result["metadata"] = {
                        "dataset": dataset_name,
                        "class_name": category,
                        "model_type": "patchcore",
                    }
                    results.append(result)
                    processed += 1

            except Exception as e:
                print(f"\nError: {e}")
                errors += len(valid_indices)

            # Handle invalid images
            invalid_count = (~valid_mask).sum().item()
            errors += invalid_count

        category_pbar.set_postfix({"done": processed, "skip": skipped, "err": errors})

        # Save periodically
        if processed % 500 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Save final results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Inference complete")
    print(f"{'='*60}")
    print(f"Processed: {processed}")
    print(f"Skipped (no model): {skipped}")
    print(f"Errors: {errors}")
    print(f"Output saved to: {output_path}")

    # Print sample output
    if results:
        print("\nSample output:")
        print(json.dumps(results[-1], indent=2))


if __name__ == "__main__":
    main()
