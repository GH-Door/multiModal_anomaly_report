#!/usr/bin/env python
"""Visualize model inference output directly.

Shows: Original | Original+GT Mask | Heatmap | Highlight
This helps debug whether bbox issues come from model output or post-processing.

Usage:
    python scripts/visualize_model_output.py \
        --checkpoint-dir output \
        --data-root /path/to/MMAD \
        --output output/model_vis.png \
        --num-samples 8
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def find_checkpoint(checkpoint_dir: Path, dataset: str, category: str) -> Path | None:
    """Find latest checkpoint for dataset/category."""
    patchcore_dir = checkpoint_dir / "Patchcore"
    if not patchcore_dir.exists():
        patchcore_dir = checkpoint_dir

    category_dir = patchcore_dir / dataset / category
    if not category_dir.exists():
        return None

    versions = []
    for v_dir in category_dir.iterdir():
        if v_dir.is_dir() and v_dir.name.startswith("v"):
            try:
                versions.append((int(v_dir.name[1:]), v_dir))
            except ValueError:
                continue

    if versions:
        latest = max(versions, key=lambda x: x[0])[1]
        ckpt = latest / "model.ckpt"
        if ckpt.exists():
            return ckpt
    return None


def find_mask_path(image_path: Path, data_root: Path) -> Path | None:
    """Find GT mask for image."""
    parts = image_path.relative_to(data_root).parts

    if "good" in parts:
        return None

    if len(parts) < 5:
        return None

    dataset = parts[0]
    category = parts[1]
    defect_type = parts[3]
    stem = image_path.stem

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


def collect_test_images(data_root: Path, datasets: list, categories: list, only_anomaly: bool = True) -> list:
    """Collect test images from datasets."""
    images = []

    for dataset in datasets:
        dataset_path = data_root / dataset
        if not dataset_path.exists():
            continue

        for cat_path in dataset_path.iterdir():
            if not cat_path.is_dir():
                continue
            if categories and cat_path.name not in categories:
                continue

            test_dir = cat_path / "test"
            if not test_dir.exists():
                continue

            for defect_dir in test_dir.iterdir():
                if not defect_dir.is_dir():
                    continue

                if only_anomaly and defect_dir.name == "good":
                    continue

                for img_file in defect_dir.iterdir():
                    if img_file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                        images.append({
                            "path": img_file,
                            "dataset": dataset,
                            "category": cat_path.name,
                            "defect_type": defect_dir.name,
                        })

    return images


def create_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    """Create colored heatmap from anomaly map."""
    # Normalize to 0-255
    if anomaly_map.max() > anomaly_map.min():
        normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        normalized = np.zeros_like(anomaly_map)

    heatmap = (normalized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color


def create_highlight(image: np.ndarray, anomaly_map: np.ndarray, threshold_ratio: float = 0.5) -> np.ndarray:
    """Create highlighted image with anomaly regions."""
    # Normalize anomaly map
    if anomaly_map.max() > anomaly_map.min():
        normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        normalized = np.zeros_like(anomaly_map)

    # Create red overlay for high anomaly regions
    threshold = normalized.max() * threshold_ratio
    mask = normalized > threshold

    highlight = image.copy()
    overlay = np.zeros_like(image)
    overlay[:, :, 2] = (normalized * 255).astype(np.uint8)  # Red channel

    # Blend
    alpha = 0.5
    highlight = cv2.addWeighted(highlight, 1 - alpha, overlay, alpha, 0)

    # Draw contour around high anomaly region
    if mask.any():
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlight, contours, -1, (0, 255, 0), 2)

    return highlight


def add_text(image: np.ndarray, text: str, position: tuple = (10, 30),
             color: tuple = (255, 255, 255), bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """Add text with background to image."""
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (position[0] - 5, position[1] - th - 5),
                  (position[0] + tw + 5, position[1] + 5), bg_color, -1)
    cv2.putText(img, text, position, font, scale, color, thickness)
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize model inference output")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/model_vis.png")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", type=str, nargs="+", default=["GoodsAD", "MVTec-LOCO"])
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=[384, 384])
    parser.add_argument("--include-normal", action="store_true", help="Include normal images")

    args = parser.parse_args()
    random.seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    data_root = Path(args.data_root)
    output_path = Path(args.output)

    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Data root: {data_root}")
    print(f"Image size: {args.image_size}")

    # Collect test images
    print(f"\nCollecting test images...")
    images = collect_test_images(
        data_root, args.datasets, args.categories,
        only_anomaly=not args.include_normal
    )
    print(f"Found {len(images)} images")

    if not images:
        print("No images found!")
        return

    # Sample
    num_samples = min(args.num_samples, len(images))
    sampled = random.sample(images, num_samples)

    # Load model
    from anomalib.models import Patchcore

    results = []
    current_model = None
    current_key = None

    for i, img_info in enumerate(sampled):
        img_path = img_info["path"]
        dataset = img_info["dataset"]
        category = img_info["category"]
        defect_type = img_info["defect_type"]

        print(f"\n[{i+1}/{num_samples}] {dataset}/{category}/{defect_type}/{img_path.name}")

        # Load model if needed
        model_key = f"{dataset}/{category}"
        if model_key != current_key:
            ckpt_path = find_checkpoint(checkpoint_dir, dataset, category)
            if ckpt_path is None:
                print(f"  Checkpoint not found!")
                continue

            print(f"  Loading model from {ckpt_path.parent.name}...")
            current_model = Patchcore.load_from_checkpoint(
                str(ckpt_path), map_location="cpu", weights_only=False
            )
            current_model.eval()
            current_key = model_key

        if current_model is None:
            continue

        # Load and preprocess image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Failed to read image!")
            continue

        original_h, original_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize for model
        h, w = args.image_size
        img_resized = cv2.resize(img_rgb, (w, h))
        img_float = img_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = current_model(tensor)

        anomaly_map = outputs.anomaly_map
        pred_score = outputs.pred_score
        pred_label = outputs.pred_label

        if anomaly_map is not None:
            # Resize anomaly map to original size
            anomaly_map = interpolate(
                anomaly_map, size=(original_h, original_w),
                mode="bilinear", align_corners=False
            )
            anomaly_map = anomaly_map[0, 0].cpu().numpy()
        else:
            anomaly_map = np.zeros((original_h, original_w))

        score = pred_score[0].item() if pred_score is not None else 0
        label = pred_label[0].item() if pred_label is not None else 0

        print(f"  Score: {score:.4f}, Label: {label}")
        print(f"  Anomaly map: min={anomaly_map.min():.4f}, max={anomaly_map.max():.4f}, mean={anomaly_map.mean():.4f}")

        # Create visualizations
        # 1. Original
        vis_original = img_bgr.copy()
        vis_original = add_text(vis_original, "Original", (10, 30))

        # 2. Original + GT Mask
        vis_mask = img_bgr.copy()
        mask_path = find_mask_path(img_path, data_root)
        if mask_path:
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask = cv2.resize(gt_mask, (original_w, original_h))
                mask_overlay = np.zeros_like(img_bgr)
                mask_overlay[:, :, 2] = gt_mask
                vis_mask = cv2.addWeighted(vis_mask, 0.7, mask_overlay, 0.3, 0)
                vis_mask = add_text(vis_mask, "GT Mask", (10, 30), (0, 0, 255))
            else:
                vis_mask = add_text(vis_mask, "GT Mask (load failed)", (10, 30), (128, 128, 128))
        else:
            if defect_type == "good":
                vis_mask = add_text(vis_mask, "Normal (no mask)", (10, 30), (0, 255, 0))
            else:
                vis_mask = add_text(vis_mask, "GT Mask (not found)", (10, 30), (128, 128, 128))

        # 3. Heatmap
        vis_heatmap = create_heatmap(anomaly_map)
        vis_heatmap = add_text(vis_heatmap, f"Heatmap (max={anomaly_map.max():.2f})", (10, 30))

        # 4. Highlight with contour
        vis_highlight = create_highlight(img_bgr, anomaly_map, threshold_ratio=0.7)
        status = "ANOMALY" if label else "NORMAL"
        color = (0, 0, 255) if label else (0, 255, 0)
        vis_highlight = add_text(vis_highlight, f"Highlight ({status}, {score:.2f})", (10, 30), color)

        # Combine horizontally
        combined = np.hstack([vis_original, vis_mask, vis_heatmap, vis_highlight])

        # Add info bar
        info_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        info_text = f"{dataset}/{category}/{defect_type}/{img_path.name}"
        cv2.putText(info_bar, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        combined = np.vstack([combined, info_bar])

        results.append(combined)

    if not results:
        print("\nNo results!")
        return

    # Resize all to same width and stack
    target_width = 1600
    resized = []
    for r in results:
        scale = target_width / r.shape[1]
        new_h = int(r.shape[0] * scale)
        resized.append(cv2.resize(r, (target_width, new_h)))

    final = np.vstack(resized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final)

    print(f"\n{'='*60}")
    print(f"Saved: {output_path}")
    print(f"Size: {final.shape[1]}x{final.shape[0]}")


if __name__ == "__main__":
    main()
