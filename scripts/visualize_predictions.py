#!/usr/bin/env python
"""Visualize AD predictions: compare bbox with GT mask.

Usage:
    # Random samples
    python scripts/visualize_predictions.py \
        --predictions ad_predictions.json \
        --data-root /path/to/MMAD \
        --output output/vis.png \
        --num-samples 12

    # Specific images
    python scripts/visualize_predictions.py \
        --predictions ad_predictions.json \
        --data-root /path/to/MMAD \
        --images 000.png 001.png

    # Save percentage per class
    python scripts/visualize_predictions.py \
        --predictions ad_predictions.json \
        --data-root /path/to/MMAD \
        --save 0.1 \
        --output output/vis_samples
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
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


def get_class_from_path(image_path: str) -> str:
    """Extract class name (dataset/category) from path."""
    rel = extract_relative_path(image_path)
    parts = Path(rel).parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return "unknown"


def infer_mask_path(image_path: str, data_root: Path) -> Path | None:
    """Infer GT mask path from image path."""
    rel_path = extract_relative_path(image_path)
    parts = Path(rel_path).parts

    if "good" in parts:
        return None

    if len(parts) < 5:
        return None

    dataset = parts[0]
    category = parts[1]
    defect_type = parts[3]
    stem = Path(parts[4]).stem

    # GoodsAD style
    mask_path = data_root / dataset / category / "ground_truth" / defect_type / f"{stem}.png"
    if mask_path.exists():
        return mask_path

    # MVTec-LOCO style (nested folder)
    mask_dir = data_root / dataset / category / "ground_truth" / defect_type / stem
    if mask_dir.exists() and mask_dir.is_dir():
        mask_files = sorted(mask_dir.glob("*.png"))
        if mask_files:
            return mask_files[0]

    # MVTec-AD style
    mask_path = data_root / dataset / category / "ground_truth" / defect_type / f"{stem}_mask.png"
    if mask_path.exists():
        return mask_path

    return None


def visualize_single(pred: dict, data_root: Path) -> np.ndarray | None:
    """Create visualization for single prediction."""
    img_path = extract_relative_path(pred.get("image_path", ""))
    full_img_path = data_root / img_path

    if not full_img_path.exists():
        return None

    img = cv2.imread(str(full_img_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    # Draw bbox on image
    img_with_bbox = img.copy()
    defect_loc = pred.get("defect_location", {})
    bbox = defect_loc.get("bbox") if defect_loc else None

    if bbox:
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

        # Draw center
        center = defect_loc.get("center")
        if center:
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(img_with_bbox, (cx, cy), 8, (0, 255, 255), -1)

    # Add score/label text
    score = pred.get("anomaly_score", pred.get("pred_score", 0))
    pred_label = pred.get("pred_label", pred.get("is_anomaly", 0))
    is_anomaly = pred_label == 1 or pred_label is True
    label = "ANOMALY" if is_anomaly else "NORMAL"
    color = (0, 0, 255) if is_anomaly else (0, 255, 0)

    cv2.putText(img_with_bbox, f"{label} ({score:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if bbox:
        region = defect_loc.get("region", "")
        cv2.putText(img_with_bbox, f"region: {region}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(img_with_bbox, "no defect", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # Load GT mask
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

    # Combine
    combined = np.hstack([img_with_bbox, mask_vis])

    # Path info
    info_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    short_path = "/".join(Path(img_path).parts[-3:])
    cv2.putText(info_bar, short_path, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    combined = np.vstack([combined, info_bar])

    return combined


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions vs GT mask")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/vis_compare.png")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-anomaly", action="store_true", help="Only sample anomalies")
    parser.add_argument("--images", type=str, nargs="+", default=None,
                        help="Specific images (filename, path, or class name)")
    parser.add_argument("--save", type=float, default=None,
                        help="Save ratio per class (e.g., 0.1 = 10%% of each class)")

    args = parser.parse_args()
    random.seed(args.seed)

    # Load predictions
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    data_root = Path(args.data_root)

    # Mode: --save (per-class percentage saving)
    if args.save is not None:
        print(f"\nSave mode: {args.save*100:.0f}% per class")

        # Group by class
        by_class = defaultdict(list)
        for p in predictions:
            cls = get_class_from_path(p.get("image_path", ""))
            by_class[cls].append(p)

        # Filter by --images if provided (class name filter)
        if args.images:
            filtered_classes = {}
            for cls, preds in by_class.items():
                for img_query in args.images:
                    if img_query in cls:
                        filtered_classes[cls] = preds
                        break
            by_class = filtered_classes
            print(f"Filtered to classes: {list(by_class.keys())}")

        # Filter anomalies if requested
        if args.only_anomaly:
            for cls in by_class:
                by_class[cls] = [p for p in by_class[cls]
                                 if p.get("is_anomaly") or p.get("pred_label") == 1]

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_saved = 0
        for cls, preds in sorted(by_class.items()):
            if not preds:
                continue

            num_save = max(1, int(len(preds) * args.save))
            sampled = random.sample(preds, min(num_save, len(preds)))

            cls_name = cls.replace("/", "_")
            print(f"\n[{cls}] Saving {len(sampled)}/{len(preds)} images...")

            for i, pred in enumerate(sampled):
                vis = visualize_single(pred, data_root)
                if vis is not None:
                    # Resize for consistency
                    scale = 800 / vis.shape[1]
                    vis = cv2.resize(vis, (800, int(vis.shape[0] * scale)))

                    filename = Path(pred.get("image_path", "")).stem
                    out_path = output_dir / f"{cls_name}_{filename}.png"
                    cv2.imwrite(str(out_path), vis)
                    total_saved += 1

        print(f"\n{'='*50}")
        print(f"Total saved: {total_saved} images to {output_dir}")
        return

    # Mode: specific images or random sampling
    # Filter by specific images if provided
    if args.images:
        filtered = []
        for p in predictions:
            img_path = p.get("image_path", "")
            rel_path = extract_relative_path(img_path)
            for img_query in args.images:
                img_query_rel = extract_relative_path(img_query)
                matched = (
                    img_query == img_path or
                    img_query_rel == rel_path or
                    img_query in img_path or
                    img_query_rel in rel_path or
                    rel_path.endswith(img_query) or
                    img_query in rel_path
                )
                if matched:
                    filtered.append(p)
                    break
        predictions = filtered
        print(f"Filtered to {len(predictions)} matching images")

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

    print(f"Processing {num_samples} samples...")

    results = []
    for i, pred in enumerate(sampled):
        vis = visualize_single(pred, data_root)
        if vis is not None:
            results.append(vis)
            img_path = extract_relative_path(pred.get("image_path", ""))
            short = "/".join(Path(img_path).parts[-3:])
            print(f"  [{i+1}/{num_samples}] {short}")

    if not results:
        print("No images processed!")
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
