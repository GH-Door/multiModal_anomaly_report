#!/usr/bin/env python
"""Visualize AD predictions: compare bbox with GT mask.

Usage:
    python scripts/visualize_predictions.py \
        --predictions ad_predictions.json \
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


def infer_mask_path(image_path: str, data_root: Path) -> Path | None:
    """Infer GT mask path from image path.

    GoodsAD: test/{defect}/xxx.jpg → ground_truth/{defect}/xxx.png
    MVTec-LOCO: test/{defect}/xxx.png → ground_truth/{defect}/xxx/000.png
    MVTec-AD: test/{defect}/xxx.png → ground_truth/{defect}/xxx_mask.png
    """
    rel_path = extract_relative_path(image_path)
    parts = Path(rel_path).parts

    # Skip normal images (no mask)
    if "good" in parts:
        return None

    # Need at least: dataset/category/test/defect_type/filename
    if len(parts) < 5:
        return None

    dataset = parts[0]
    category = parts[1]
    defect_type = parts[3]
    filename = parts[4]
    stem = Path(filename).stem

    # Try different mask path patterns
    candidates = []

    # GoodsAD: ground_truth/{defect}/xxx.png
    candidates.append(
        data_root / dataset / category / "ground_truth" / defect_type / f"{stem}.png"
    )

    # MVTec-LOCO: ground_truth/{defect}/xxx/000.png (nested folder)
    mask_dir = data_root / dataset / category / "ground_truth" / defect_type / stem
    if mask_dir.exists() and mask_dir.is_dir():
        mask_files = sorted(mask_dir.glob("*.png"))
        if mask_files:
            candidates.insert(0, mask_files[0])  # Priority

    # MVTec-AD: ground_truth/{defect}/xxx_mask.png
    candidates.append(
        data_root / dataset / category / "ground_truth" / defect_type / f"{stem}_mask.png"
    )

    # Return first existing
    for c in candidates:
        if c.exists():
            return c

    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions vs GT mask")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/vis_compare.png")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-anomaly", action="store_true", help="Only sample anomalies")

    args = parser.parse_args()
    random.seed(args.seed)

    # Load predictions
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # Debug: check fields
    if predictions:
        print(f"Fields: {list(predictions[0].keys())}")

    # Filter anomalies if requested
    if args.only_anomaly:
        predictions = [
            p for p in predictions
            if p.get("pred_label") == 1 or p.get("is_anomaly") == True
        ]
        print(f"Filtered to {len(predictions)} anomalies")

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
            print(f"Failed to read: {full_img_path}")
            continue

        h, w = img.shape[:2]

        # Draw bbox on image
        img_with_bbox = img.copy()
        defect_loc = pred.get("defect_location", {})
        bbox = defect_loc.get("bbox") if defect_loc else None

        if bbox:
            x_min, y_min, x_max, y_max = bbox
            # Clip to image bounds
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

            # Draw center
            center = defect_loc.get("center")
            if center:
                cx, cy = int(center[0] * w), int(center[1] * h)
                cv2.circle(img_with_bbox, (cx, cy), 8, (0, 255, 255), -1)

        # Add score/label text
        score = pred.get("anomaly_score", pred.get("pred_score", 0))
        pred_label = pred.get("pred_label", pred.get("is_anomaly", 0))
        is_anomaly = pred_label == 1 or pred_label is True
        label = "ANOMALY" if is_anomaly else "NORMAL"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        cv2.putText(img_with_bbox, f"{label} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # bbox info
        if bbox:
            region = defect_loc.get("region", "")
            cv2.putText(img_with_bbox, f"region: {region}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(img_with_bbox, "no bbox", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Load GT mask - first try from JSON, then infer from path
        mask_path = pred.get("mask_path")
        if mask_path:
            mask_path = data_root / extract_relative_path(mask_path)
        else:
            mask_path = infer_mask_path(pred.get("image_path", ""), data_root)

        mask_vis = None
        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (w, h))
                mask_color = np.zeros_like(img)
                mask_color[:, :, 2] = mask
                mask_vis = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)
                cv2.putText(mask_vis, "GT MASK", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if mask_vis is None:
            if "/good/" in img_path:
                mask_vis = img.copy()
                cv2.putText(mask_vis, "NORMAL (no defect)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                mask_vis = np.zeros_like(img)
                cv2.putText(mask_vis, "MASK NOT FOUND", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                print(f"  Mask not found for: {img_path}")

        # Combine
        combined = np.hstack([img_with_bbox, mask_vis])

        # Path info
        info_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        short_path = "/".join(Path(img_path).parts[-3:])
        cv2.putText(info_bar, short_path, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        combined = np.vstack([combined, info_bar])

        results.append(combined)
        mask_status = "found" if (mask_path and mask_path.exists()) else "not found"
        print(f"Processed: {short_path} | {label} | bbox={'yes' if bbox else 'no'} | mask={mask_status}")

    if not results:
        print("No images processed")
        return

    # Resize and stack
    target_width = 800
    resized = []
    for r in results:
        scale = target_width / r.shape[1]
        new_h = int(r.shape[0] * scale)
        resized.append(cv2.resize(r, (target_width, new_h)))

    final = np.vstack(resized)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final)

    print(f"\nSaved: {output_path}")
    print(f"Size: {final.shape[1]}x{final.shape[0]}")


if __name__ == "__main__":
    main()
