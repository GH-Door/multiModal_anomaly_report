#!/usr/bin/env python
"""Visualize AD inference predictions on original images.

Randomly samples predictions and draws bbox, center, region, score etc.

Usage:
    python scripts/visualize_predictions.py \
        --predictions output/ad_predictions.json \
        --data-root /path/to/MMAD \
        --output output/visualizations \
        --num-samples 20
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def draw_prediction(
    image: np.ndarray,
    pred: dict,
    show_bbox: bool = True,
    show_center: bool = True,
    show_info: bool = True,
) -> np.ndarray:
    """Draw prediction info on image.

    Args:
        image: BGR image
        pred: Prediction dict with anomaly_score, is_anomaly, defect_location, etc.
        show_bbox: Draw bounding box
        show_center: Draw center point
        show_info: Draw text info

    Returns:
        Annotated image
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Colors
    COLOR_NORMAL = (0, 255, 0)      # Green
    COLOR_ANOMALY = (0, 0, 255)     # Red
    COLOR_BBOX = (255, 0, 255)      # Magenta
    COLOR_CENTER = (255, 255, 0)    # Cyan
    COLOR_TEXT_BG = (0, 0, 0)       # Black

    # Determine if anomaly
    is_anomaly = pred.get("is_anomaly", False)
    anomaly_score = pred.get("anomaly_score", pred.get("pred_score", 0))
    pred_label = pred.get("pred_label")

    # If pred_label exists, use it
    if pred_label is not None:
        is_anomaly = bool(pred_label)

    main_color = COLOR_ANOMALY if is_anomaly else COLOR_NORMAL

    # Draw defect location if exists
    defect_loc = pred.get("defect_location", {})

    if defect_loc.get("has_defect") and show_bbox:
        bbox = defect_loc.get("bbox")
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLOR_BBOX, 2)

    if defect_loc.get("center") and show_center:
        center = defect_loc.get("center")
        # center is normalized [0-1]
        cx = int(center[0] * w)
        cy = int(center[1] * h)
        cv2.circle(img, (cx, cy), 8, COLOR_CENTER, -1)
        cv2.circle(img, (cx, cy), 10, COLOR_CENTER, 2)

    # Draw info text
    if show_info:
        # Status bar at top
        status = "ANOMALY" if is_anomaly else "NORMAL"
        score_text = f"{status} (score: {anomaly_score:.3f})"

        # Background for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(score_text, font, font_scale, thickness)

        cv2.rectangle(img, (0, 0), (text_w + 20, text_h + 20), COLOR_TEXT_BG, -1)
        cv2.putText(img, score_text, (10, text_h + 10), font, font_scale, main_color, thickness)

        # Additional info
        y_offset = text_h + 40
        info_lines = []

        # Region
        if defect_loc.get("region"):
            info_lines.append(f"Region: {defect_loc['region']}")

        # Area ratio
        if defect_loc.get("area_ratio"):
            info_lines.append(f"Area: {defect_loc['area_ratio']*100:.1f}%")

        # Confidence (from defect_location)
        if defect_loc.get("confidence"):
            info_lines.append(f"Conf: {defect_loc['confidence']:.3f}")

        # Confidence level (new field)
        confidence_info = pred.get("confidence", {})
        if confidence_info:
            level = confidence_info.get("level", "")
            reliability = confidence_info.get("reliability", "")
            if level:
                info_lines.append(f"Level: {level}")
            if reliability:
                info_lines.append(f"Reliability: {reliability}")

        # Map stats
        map_stats = pred.get("map_stats", {})
        if map_stats:
            info_lines.append(f"Map max: {map_stats.get('max', 0):.3f}")

        for line in info_lines:
            (line_w, line_h), _ = cv2.getTextSize(line, font, 0.5, 1)
            cv2.rectangle(img, (0, y_offset - line_h - 5), (line_w + 20, y_offset + 5), COLOR_TEXT_BG, -1)
            cv2.putText(img, line, (10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += line_h + 15

    # Border based on prediction
    border_thickness = 5
    cv2.rectangle(img, (0, 0), (w-1, h-1), main_color, border_thickness)

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize AD predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of images")
    parser.add_argument("--output", type=str, default="output/visualizations", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of random samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--filter-anomaly", action="store_true", help="Only show anomalies")
    parser.add_argument("--filter-normal", action="store_true", help="Only show normals")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--no-bbox", action="store_true", help="Don't draw bbox")
    parser.add_argument("--no-center", action="store_true", help="Don't draw center")
    parser.add_argument("--grid", action="store_true", help="Create grid image instead of individual files")
    parser.add_argument("--grid-cols", type=int, default=4, help="Grid columns")
    parser.add_argument("--resize", type=int, default=None, help="Resize images to this size")

    args = parser.parse_args()

    random.seed(args.seed)

    # Load predictions
    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        print(f"Error: Predictions file not found: {predictions_path}")
        return

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # Filter predictions
    filtered = predictions

    if args.filter_anomaly:
        filtered = [p for p in filtered if p.get("is_anomaly") or p.get("pred_label") == 1]
        print(f"Filtered to {len(filtered)} anomalies")
    elif args.filter_normal:
        filtered = [p for p in filtered if not p.get("is_anomaly") and p.get("pred_label", 1) == 0]
        print(f"Filtered to {len(filtered)} normals")

    if args.category:
        filtered = [p for p in filtered if args.category in p.get("image_path", "")]
        print(f"Filtered to {len(filtered)} for category '{args.category}'")

    if not filtered:
        print("No predictions after filtering")
        return

    # Sample
    num_samples = min(args.num_samples, len(filtered))
    sampled = random.sample(filtered, num_samples)
    print(f"Sampled {num_samples} predictions")

    # Data root
    data_root = Path(args.data_root)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    images = []
    valid_preds = []

    for i, pred in enumerate(sampled):
        image_path = pred.get("image_path", "")

        # Handle absolute vs relative paths
        if image_path.startswith("/"):
            # Absolute path - try to extract relative part
            # e.g., /content/drive/.../MMAD/GoodsAD/... -> GoodsAD/...
            for dataset in ["GoodsAD", "MVTec-AD", "MVTec-LOCO", "VisA"]:
                if dataset in image_path:
                    idx = image_path.index(dataset)
                    image_path = image_path[idx:]
                    break

        full_path = data_root / image_path

        if not full_path.exists():
            print(f"  [{i+1}] Image not found: {full_path}")
            continue

        img = cv2.imread(str(full_path))
        if img is None:
            print(f"  [{i+1}] Failed to read: {full_path}")
            continue

        # Resize if specified
        if args.resize:
            img = cv2.resize(img, (args.resize, args.resize))

        # Draw prediction
        annotated = draw_prediction(
            img, pred,
            show_bbox=not args.no_bbox,
            show_center=not args.no_center,
        )

        images.append(annotated)
        valid_preds.append(pred)

        # Save individual if not grid mode
        if not args.grid:
            # Create filename from path
            path_parts = Path(image_path).parts
            if len(path_parts) >= 3:
                filename = f"{path_parts[-4]}_{path_parts[-3]}_{path_parts[-1]}"
            else:
                filename = Path(image_path).name

            is_anomaly = pred.get("is_anomaly") or pred.get("pred_label") == 1
            prefix = "anomaly" if is_anomaly else "normal"
            output_path = output_dir / f"{prefix}_{i:03d}_{filename}"

            cv2.imwrite(str(output_path), annotated)
            print(f"  [{i+1}] Saved: {output_path.name}")

    print(f"\nProcessed {len(images)} images")

    # Create grid if requested
    if args.grid and images:
        cols = args.grid_cols
        rows = (len(images) + cols - 1) // cols

        # Resize all to same size
        target_size = args.resize or 300
        resized = [cv2.resize(img, (target_size, target_size)) for img in images]

        # Pad with black if needed
        while len(resized) < rows * cols:
            resized.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))

        # Create grid
        grid_rows = []
        for r in range(rows):
            row_imgs = resized[r * cols : (r + 1) * cols]
            grid_rows.append(np.hstack(row_imgs))

        grid = np.vstack(grid_rows)

        grid_path = output_dir / "grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"\nGrid saved: {grid_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    anomaly_count = sum(1 for p in valid_preds if p.get("is_anomaly") or p.get("pred_label") == 1)
    normal_count = len(valid_preds) - anomaly_count

    print(f"Total visualized: {len(valid_preds)}")
    print(f"  Anomalies: {anomaly_count}")
    print(f"  Normals: {normal_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
