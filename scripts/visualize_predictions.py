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


def infer_mask_path(image_path: str, data_root: Path, debug: bool = False) -> Path | None:
    """Infer GT mask path from image path.

    GoodsAD: test/{defect}/xxx.jpg → ground_truth/{defect}/xxx.png
    MVTec-LOCO: test/{defect}/xxx.png → ground_truth/{defect}/xxx/000.png
    """
    rel_path = extract_relative_path(image_path)
    parts = Path(rel_path).parts

    if debug:
        print(f"    [DEBUG] rel_path: {rel_path}")
        print(f"    [DEBUG] parts: {parts}")

    # Skip normal images (no mask)
    if "good" in parts:
        if debug:
            print(f"    [DEBUG] Skipping normal image (good)")
        return None

    # Need at least: dataset/category/test/defect_type/filename
    if len(parts) < 5:
        if debug:
            print(f"    [DEBUG] Not enough parts: {len(parts)}")
        return None

    dataset = parts[0]
    category = parts[1]
    defect_type = parts[3]
    filename = parts[4]
    stem = Path(filename).stem

    if debug:
        print(f"    [DEBUG] dataset={dataset}, category={category}, defect_type={defect_type}, stem={stem}")

    # Try different mask path patterns
    candidates = []

    # Pattern 1: GoodsAD style - ground_truth/{defect}/xxx.png
    candidates.append(
        data_root / dataset / category / "ground_truth" / defect_type / f"{stem}.png"
    )

    # Pattern 2: MVTec-LOCO style - ground_truth/{defect}/xxx/000.png (nested folder)
    mask_dir = data_root / dataset / category / "ground_truth" / defect_type / stem
    if mask_dir.exists() and mask_dir.is_dir():
        mask_files = sorted(mask_dir.glob("*.png"))
        if mask_files:
            candidates.insert(0, mask_files[0])

    # Pattern 3: MVTec-AD style - ground_truth/{defect}/xxx_mask.png
    candidates.append(
        data_root / dataset / category / "ground_truth" / defect_type / f"{stem}_mask.png"
    )

    if debug:
        print(f"    [DEBUG] Candidates:")
        for c in candidates:
            print(f"      - {c} (exists: {c.exists()})")

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
    parser.add_argument("--images", type=str, nargs="+", default=None, help="Specific image filenames to visualize")
    parser.add_argument("--debug", action="store_true", help="Print debug info")

    args = parser.parse_args()
    random.seed(args.seed)

    # Load predictions
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")
    print(f"Data root: {args.data_root}")

    # Debug: check fields and sample
    if predictions:
        print(f"Fields: {list(predictions[0].keys())}")
        print(f"Sample image_path: {predictions[0].get('image_path', 'N/A')[:100]}")

    # Filter by specific images if provided
    if args.images:
        filtered = []
        for p in predictions:
            img_path = p.get("image_path", "")
            for img_name in args.images:
                if img_name in img_path:
                    filtered.append(p)
                    break
        predictions = filtered
        print(f"Filtered to {len(predictions)} matching images: {args.images}")

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

    # Sample (skip if specific images provided)
    if args.images:
        sampled = predictions
    else:
        num_samples = min(args.num_samples, len(predictions))
        sampled = random.sample(predictions, num_samples)

    data_root = Path(args.data_root)

    # Verify data_root exists
    if not data_root.exists():
        print(f"ERROR: data_root does not exist: {data_root}")
        return

    results = []

    for i, pred in enumerate(sampled):
        print(f"\n[{i+1}/{num_samples}] Processing...")

        # Load original image
        img_path = extract_relative_path(pred.get("image_path", ""))
        full_img_path = data_root / img_path

        print(f"  Image: {img_path}")
        if args.debug:
            print(f"  Full path: {full_img_path}")

        if not full_img_path.exists():
            print(f"  ERROR: Image not found!")
            continue

        img = cv2.imread(str(full_img_path))
        if img is None:
            print(f"  ERROR: Failed to read image!")
            continue

        h, w = img.shape[:2]
        print(f"  Size: {w}x{h}")

        # Draw bbox on image
        img_with_bbox = img.copy()
        defect_loc = pred.get("defect_location", {})
        bboxes = defect_loc.get("bboxes", []) if defect_loc else []
        bbox = defect_loc.get("bbox") if defect_loc else None

        # Draw all bboxes (secondary ones in different color)
        for i, bb in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bb
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            # Primary bbox in red, secondary in orange
            color = (0, 0, 255) if i == 0 else (0, 165, 255)
            thickness = 3 if i == 0 else 2
            cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), color, thickness)

        if bboxes:
            print(f"  BBoxes: {len(bboxes)} detected")

        # Draw center of primary bbox
        center = defect_loc.get("center") if defect_loc else None
        if center:
            # Check if center is pixel coords or normalized
            cx, cy = center
            if isinstance(cx, float) and cx <= 1:
                cx, cy = int(cx * w), int(cy * h)
            else:
                cx, cy = int(cx), int(cy)
            cv2.circle(img_with_bbox, (cx, cy), 8, (0, 255, 255), -1)

        # Add score/label text
        score = pred.get("anomaly_score", pred.get("pred_score", 0))
        pred_label = pred.get("pred_label", pred.get("is_anomaly", 0))
        is_anomaly = pred_label == 1 or pred_label is True
        label = "ANOMALY" if is_anomaly else "NORMAL"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        cv2.putText(img_with_bbox, f"{label} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if bboxes:
            region = defect_loc.get("region", "")
            num_defects = defect_loc.get("num_defects", len(bboxes))
            cv2.putText(img_with_bbox, f"region: {region} ({num_defects} defects)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(img_with_bbox, "no defect", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Find GT mask
        mask_path = infer_mask_path(pred.get("image_path", ""), data_root, debug=args.debug)

        mask_vis = None
        if mask_path and mask_path.exists():
            print(f"  Mask: FOUND - {mask_path.name}")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (w, h))
                mask_color = np.zeros_like(img)
                mask_color[:, :, 2] = mask
                mask_vis = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)
                cv2.putText(mask_vis, "GT MASK", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            print(f"  Mask: NOT FOUND")

        if mask_vis is None:
            if "/good/" in img_path:
                mask_vis = img.copy()
                cv2.putText(mask_vis, "NORMAL (no defect)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                mask_vis = np.zeros_like(img)
                cv2.putText(mask_vis, "MASK NOT FOUND", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        # Combine
        combined = np.hstack([img_with_bbox, mask_vis])

        # Path info
        info_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        short_path = "/".join(Path(img_path).parts[-3:])
        cv2.putText(info_bar, short_path, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        combined = np.vstack([combined, info_bar])

        results.append(combined)

    if not results:
        print("\nNo images processed!")
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

    print(f"\n{'='*50}")
    print(f"Saved: {output_path}")
    print(f"Size: {final.shape[1]}x{final.shape[0]}")


if __name__ == "__main__":
    main()
