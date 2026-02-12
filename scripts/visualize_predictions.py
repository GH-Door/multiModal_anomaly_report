#!/usr/bin/env python
"""Visualize AD predictions: compare bbox with GT mask.

Usage:
    python scripts/visualize_predictions.py \
        --predictions predictions.json \
        --data-root /path/to/MMAD \
        --output output/vis_compare.png \
        --num-samples 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def extract_relative_path(image_path: str) -> str:
    """Extract relative path from absolute path."""
    for dataset in ["GoodsAD", "MVTec-AD", "MVTec-LOCO", "VisA"]:
        if dataset in image_path:
            idx = image_path.index(dataset)
            return image_path[idx:]
    return image_path


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions vs GT mask")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/vis_compare.png")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-anomaly", action="store_true", help="Only sample anomalies (with mask)")

    args = parser.parse_args()
    random.seed(args.seed)

    # Load predictions
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # Filter: only those with mask_path (anomalies with GT)
    if args.only_anomaly:
        predictions = [p for p in predictions if p.get("mask_path")]
        print(f"Filtered to {len(predictions)} with GT mask")

    if not predictions:
        print("No predictions to visualize")
        return

    # Sample
    num_samples = min(args.num_samples, len(predictions))
    sampled = random.sample(predictions, num_samples)

    data_root = Path(args.data_root)
    results = []

    for pred in sampled:
        # Load original image
        img_path = extract_relative_path(pred.get("image_path", ""))
        full_img_path = data_root / img_path

        if not full_img_path.exists():
            print(f"Image not found: {full_img_path}")
            continue

        img = cv2.imread(str(full_img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Draw bbox on image
        img_with_bbox = img.copy()
        defect_loc = pred.get("defect_location", {})
        bbox = defect_loc.get("bbox")

        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
            # Draw center
            center = defect_loc.get("center")
            if center:
                cx, cy = int(center[0] * w), int(center[1] * h)
                cv2.circle(img_with_bbox, (cx, cy), 8, (0, 255, 255), -1)

        # Add score text
        score = pred.get("anomaly_score", pred.get("pred_score", 0))
        is_anomaly = pred.get("is_anomaly", pred.get("pred_label", 0))
        label = "ANOMALY" if is_anomaly else "NORMAL"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        cv2.putText(img_with_bbox, f"{label} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Load GT mask
        mask_path = pred.get("mask_path")
        if mask_path:
            mask_path = extract_relative_path(mask_path)
            full_mask_path = data_root / mask_path

            if full_mask_path.exists():
                mask = cv2.imread(str(full_mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Resize mask to match image
                    mask = cv2.resize(mask, (w, h))
                    # Convert to color (red overlay)
                    mask_color = np.zeros_like(img)
                    mask_color[:, :, 2] = mask  # Red channel
                    # Blend with original
                    mask_vis = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)
                    cv2.putText(mask_vis, "GT MASK", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    mask_vis = np.zeros_like(img)
                    cv2.putText(mask_vis, "MASK LOAD FAILED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                mask_vis = np.zeros_like(img)
                cv2.putText(mask_vis, "NO MASK FILE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        else:
            # Normal image - no mask
            mask_vis = np.zeros_like(img)
            cv2.putText(mask_vis, "NORMAL (no mask)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Combine: [image + bbox] | [GT mask]
        combined = np.hstack([img_with_bbox, mask_vis])

        # Add path info at bottom
        info_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        short_path = "/".join(Path(img_path).parts[-3:])
        cv2.putText(info_bar, short_path, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        combined = np.vstack([combined, info_bar])

        results.append(combined)
        print(f"Processed: {short_path}")

    if not results:
        print("No images processed")
        return

    # Resize all to same width
    target_width = 800
    resized = []
    for r in results:
        scale = target_width / r.shape[1]
        new_h = int(r.shape[0] * scale)
        resized.append(cv2.resize(r, (target_width, new_h)))

    # Stack vertically
    final = np.vstack(resized)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final)

    print(f"\nSaved: {output_path}")
    print(f"Image size: {final.shape[1]}x{final.shape[0]}")


if __name__ == "__main__":
    main()
