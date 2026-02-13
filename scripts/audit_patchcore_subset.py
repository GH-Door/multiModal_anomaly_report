#!/usr/bin/env python
"""Run PatchCore ckpt inference on a small subset and audit bbox-vs-mask quality.

This script does all of the following in one run:
1. Sample a small number of test images per class from MMAD json.
2. Run checkpoint-based PatchCore inference.
3. Save prediction json (including bbox from anomaly map thresholding).
4. Save visual checks per sample:
   - original
   - mask overlay
   - heatmap
   - highlight
   - bbox vs mask
5. Compute bbox/mask similarity metrics and save csv/json summaries.

Example:
    python scripts/audit_patchcore_subset.py \
        --checkpoint-dir /path/to/checkpoints \
        --data-root /Volumes/T7/Dataset/MMAD \
        --mmad-json /Volumes/T7/Dataset/MMAD/mmad_10classes.json \
        --samples-per-class 4 \
        --datasets GoodsAD MVTec-LOCO \
        --output-dir output/audit_subset \
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


DATASET_MARKERS = ("GoodsAD", "MVTec-LOCO", "MVTec-AD", "DS-MVTec", "VisA")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _is_hub_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keys = [
        "locate the file on the hub",
        "hf_hub_offline",
        "huggingface.co",
        "local cache",
    ]
    return any(k in msg for k in keys)


def normalize_rel_path(path_like: str) -> str:
    """Convert absolute-like image path to dataset-relative path."""
    path_like = path_like.strip()
    normalized = path_like.replace("\\", "/")
    for marker in DATASET_MARKERS:
        key = f"{marker}/"
        if key in normalized:
            return normalized[normalized.index(key):]
    return normalized.lstrip("/")


def split_rel_image_path(rel_path: str) -> Optional[Tuple[str, str, str, str, str]]:
    parts = Path(rel_path).parts
    if len(parts) < 5:
        return None
    dataset, category, split, defect_type = parts[0], parts[1], parts[2], parts[3]
    filename = parts[-1]
    return dataset, category, split, defect_type, filename


def compute_defect_location(anomaly_map: np.ndarray, threshold: float) -> Dict[str, Any]:
    h, w = anomaly_map.shape
    defect_mask = anomaly_map > float(threshold)

    if not defect_mask.any():
        return {
            "has_defect": False,
            "region": "none",
            "bbox": None,
            "center": None,
            "area_ratio": 0.0,
        }

    coords = np.where(defect_mask)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    center_y = (y_min + y_max) / 2 / h
    center_x = (x_min + x_max) / 2 / w
    region_y = "top" if center_y < 1 / 3 else ("bottom" if center_y > 2 / 3 else "middle")
    region_x = "left" if center_x < 1 / 3 else ("right" if center_x > 2 / 3 else "center")
    region = "center" if (region_y == "middle" and region_x == "center") else f"{region_y}-{region_x}"

    area_ratio = float(defect_mask.sum()) / float(h * w)
    confidence = float(anomaly_map[defect_mask].max())

    return {
        "has_defect": True,
        "region": region,
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "center": [round(float(center_x), 3), round(float(center_y), 3)],
        "area_ratio": round(area_ratio, 4),
        "confidence": round(confidence, 4),
    }


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    min_v = float(anomaly_map.min())
    max_v = float(anomaly_map.max())
    if max_v <= min_v:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return (anomaly_map - min_v) / (max_v - min_v)


def make_heatmap(anomaly_map: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    norm_map = normalize_map(anomaly_map)
    heat_u8 = (norm_map * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.resize(heat, size)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.4) -> np.ndarray:
    out = image.copy()
    binary = mask > 0
    if not np.any(binary):
        return out
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[binary] = color
    return cv2.addWeighted(out, 1.0, overlay, alpha, 0.0)


def draw_bbox(image: np.ndarray, bbox: Optional[List[int]], color: Tuple[int, int, int], label: str = "") -> np.ndarray:
    out = image.copy()
    if bbox is None:
        return out
    h, w = out.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    if x1 <= x0 or y1 <= y0:
        return out
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    if label:
        cv2.putText(out, label, (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_mask_contour(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    binary = (mask > 0).astype(np.uint8) * 255
    if binary.max() == 0:
        return out
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def add_title(panel: np.ndarray, title: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


def to_panel(image: np.ndarray, panel_size: int) -> np.ndarray:
    return cv2.resize(image, (panel_size, panel_size), interpolation=cv2.INTER_AREA)


def bbox_to_mask(shape: Tuple[int, int], bbox: Optional[List[int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if bbox is None:
        return mask
    h, w = shape
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    if x1 <= x0 or y1 <= y0:
        return mask
    mask[y0:y1 + 1, x0:x1 + 1] = 1
    return mask


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def iou_from_binary(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a_bin = a > 0
    b_bin = b > 0
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return None
    inter = np.logical_and(a_bin, b_bin).sum()
    return float(inter / union)


def compute_bbox_mask_metrics(
    pred_bbox: Optional[List[int]],
    gt_mask: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    if gt_mask is None:
        return {
            "box_iou": None,
            "bbox_mask_iou": None,
            "bbox_mask_precision": None,
            "bbox_mask_recall": None,
        }

    gt_bin = (gt_mask > 0).astype(np.uint8)
    gt_bbox = mask_to_bbox(gt_bin)
    pred_rect = bbox_to_mask(gt_bin.shape, pred_bbox)

    box_iou = None
    if pred_bbox is not None and gt_bbox is not None:
        pred_box_mask = bbox_to_mask(gt_bin.shape, pred_bbox)
        gt_box_mask = bbox_to_mask(gt_bin.shape, gt_bbox)
        box_iou = iou_from_binary(pred_box_mask, gt_box_mask)

    bbox_mask_iou = iou_from_binary(pred_rect, gt_bin)

    pred_area = int(pred_rect.sum())
    gt_area = int(gt_bin.sum())
    inter = int(np.logical_and(pred_rect > 0, gt_bin > 0).sum())

    precision = float(inter / pred_area) if pred_area > 0 else None
    recall = float(inter / gt_area) if gt_area > 0 else None

    return {
        "box_iou": box_iou,
        "bbox_mask_iou": bbox_mask_iou,
        "bbox_mask_precision": precision,
        "bbox_mask_recall": recall,
    }


@dataclass
class SampleRecord:
    rel_path: str
    dataset: str
    category: str
    defect_type: str
    metadata: Dict[str, Any]


class PatchCoreCkptRunner:
    """Lightweight checkpoint runner for per-class PatchCore inference."""

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        version: Optional[int],
        device: str,
        input_size: Tuple[int, int],
        allow_online_backbone: bool,
        default_threshold: float,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.version = version
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.input_size = input_size
        self.allow_online_backbone = allow_online_backbone
        self.default_threshold = default_threshold
        self._models: Dict[str, Any] = {}
        self._warmed_up: set[str] = set()

    def _find_checkpoint(self, dataset: str, category: str) -> Optional[Path]:
        patchcore_dir = self.checkpoint_dir / "Patchcore"
        if not patchcore_dir.exists():
            patchcore_dir = self.checkpoint_dir

        category_dir = patchcore_dir / dataset / category
        if not category_dir.exists():
            return None

        if self.version is not None:
            ckpt = category_dir / f"v{self.version}" / "model.ckpt"
            return ckpt if ckpt.exists() else None

        versions: List[Tuple[int, Path]] = []
        for v_dir in category_dir.iterdir():
            if v_dir.is_dir() and v_dir.name.startswith("v"):
                try:
                    versions.append((int(v_dir.name[1:]), v_dir))
                except ValueError:
                    continue
        if not versions:
            return None
        latest = max(versions, key=lambda x: x[0])[1]
        ckpt = latest / "model.ckpt"
        return ckpt if ckpt.exists() else None

    @staticmethod
    def _load_from_checkpoint(ckpt_path: Path, *, pre_trained_override: Optional[bool]):
        from anomalib.models import Patchcore

        kwargs: Dict[str, Any] = {"map_location": "cpu", "weights_only": False}
        if pre_trained_override is not None:
            kwargs["pre_trained"] = pre_trained_override
        try:
            return Patchcore.load_from_checkpoint(str(ckpt_path), **kwargs)
        except TypeError as exc:
            if "pre_trained" in str(exc):
                kwargs.pop("pre_trained", None)
                return Patchcore.load_from_checkpoint(str(ckpt_path), **kwargs)
            raise

    @staticmethod
    def _load_manual_no_hub(ckpt_path: Path):
        from anomalib.models import Patchcore

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        hyper = ckpt.get("hyper_parameters", {})

        sig = inspect.signature(Patchcore.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        init_kwargs = {k: v for k, v in hyper.items() if k in allowed}
        init_kwargs["pre_trained"] = False

        model = Patchcore(**init_kwargs)
        model.load_state_dict(state_dict, strict=False)
        return model

    def _load_model(self, dataset: str, category: str):
        key = f"{dataset}/{category}"
        if key in self._models:
            return self._models[key]

        ckpt_path = self._find_checkpoint(dataset, category)
        if ckpt_path is None:
            raise FileNotFoundError(f"Checkpoint not found for {key}")

        try:
            model = self._load_from_checkpoint(ckpt_path, pre_trained_override=False)
        except Exception as exc:
            if not _is_hub_error(exc):
                raise
            try:
                model = self._load_manual_no_hub(ckpt_path)
            except Exception as manual_exc:
                if not self.allow_online_backbone:
                    raise RuntimeError(
                        f"Failed to load {key} due to Hub resolution; re-run with --allow-online-backbone"
                    ) from manual_exc
                model = self._load_from_checkpoint(ckpt_path, pre_trained_override=None)

        model.eval()
        model.to(self.device)
        self._models[key] = model
        return model

    def warmup(self, dataset: str, category: str) -> None:
        key = f"{dataset}/{category}"
        if key in self._warmed_up:
            return
        model = self._load_model(dataset, category)
        dummy = torch.randn(1, 3, self.input_size[0], self.input_size[1], device=self.device)
        with torch.inference_mode():
            _ = model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._warmed_up.add(key)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        h, w = self.input_size
        img = cv2.resize(image, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, dataset: str, category: str, image: np.ndarray) -> Dict[str, Any]:
        model = self._load_model(dataset, category)
        tensor = self._preprocess(image)
        original_size = image.shape[:2]

        with torch.inference_mode():
            outputs = model(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        anomaly_map = getattr(outputs, "anomaly_map", None)
        pred_score = getattr(outputs, "pred_score", None)
        pred_label = getattr(outputs, "pred_label", None)

        if anomaly_map is None:
            raise RuntimeError("Model output does not include anomaly_map")

        if anomaly_map.shape[-2:] != (original_size[0], original_size[1]):
            anomaly_map = interpolate(anomaly_map, size=original_size, mode="bilinear", align_corners=False)
        anomaly_map_np = anomaly_map[0, 0].detach().cpu().numpy()

        score = float(pred_score[0].detach().cpu()) if pred_score is not None else float(np.max(anomaly_map_np))
        pred_is_anomaly = bool(pred_label[0].detach().cpu()) if pred_label is not None else (score > self.default_threshold)

        threshold_used = self.default_threshold
        post = getattr(model, "post_processor", None)
        if post is not None and hasattr(post, "normalized_image_threshold"):
            thr = post.normalized_image_threshold
            if isinstance(thr, torch.Tensor):
                threshold_used = float(thr.detach().cpu())
            else:
                threshold_used = float(thr)

        return {
            "anomaly_map": anomaly_map_np,
            "anomaly_score": score,
            "is_anomaly": pred_is_anomaly,
            "threshold": threshold_used,
        }

    def clear(self) -> None:
        self._models.clear()
        self._warmed_up.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def load_mmad_entries(mmad_json: Path) -> Dict[str, Dict[str, Any]]:
    with open(mmad_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("mmad json must be a dictionary keyed by image path")
    return data


def sample_records(
    mmad_entries: Dict[str, Dict[str, Any]],
    *,
    datasets: Optional[List[str]],
    categories: Optional[List[str]],
    include_good: bool,
    samples_per_class: int,
    seed: int,
) -> List[SampleRecord]:
    random.seed(seed)

    grouped: Dict[Tuple[str, str], List[SampleRecord]] = defaultdict(list)
    for key, meta in mmad_entries.items():
        rel = normalize_rel_path(key)
        parsed = split_rel_image_path(rel)
        if parsed is None:
            continue
        dataset, category, split, defect_type, filename = parsed
        if split != "test":
            continue
        if Path(filename).suffix.lower() not in IMAGE_EXTS:
            continue
        if datasets and dataset not in datasets:
            continue
        if categories and category not in categories:
            continue
        if not include_good and defect_type == "good":
            continue
        grouped[(dataset, category)].append(
            SampleRecord(
                rel_path=rel,
                dataset=dataset,
                category=category,
                defect_type=defect_type,
                metadata=meta if isinstance(meta, dict) else {},
            )
        )

    sampled: List[SampleRecord] = []
    for key, recs in sorted(grouped.items()):
        if not recs:
            continue
        k = min(samples_per_class, len(recs))
        sampled.extend(random.sample(recs, k))
    return sampled


def resolve_image_path(sample: SampleRecord, data_root: Path) -> Optional[Path]:
    local = data_root / sample.rel_path
    if local.exists():
        return local

    meta_path = sample.metadata.get("image_path")
    if isinstance(meta_path, str) and meta_path:
        rel = normalize_rel_path(meta_path)
        local2 = data_root / rel
        if local2.exists():
            return local2
    return None


def _try_mask_path_from_meta(sample: SampleRecord, data_root: Path) -> Optional[Path]:
    value = sample.metadata.get("mask_path")
    if not isinstance(value, str) or not value:
        return None
    rel = normalize_rel_path(value)
    p = data_root / rel
    return p if p.exists() else None


def resolve_mask_ref(sample: SampleRecord, data_root: Path) -> Optional[Path]:
    if sample.defect_type == "good":
        return None

    from_meta = _try_mask_path_from_meta(sample, data_root)
    if from_meta is not None:
        return from_meta

    stem = Path(sample.rel_path).stem
    gt_root = data_root / sample.dataset / sample.category / "ground_truth" / sample.defect_type

    # GoodsAD style: .../ground_truth/<defect>/<stem>.png
    cand = gt_root / f"{stem}.png"
    if cand.exists():
        return cand

    # MVTec-LOCO style: .../ground_truth/<defect>/<stem>/*.png
    cand_dir = gt_root / stem
    if cand_dir.exists() and cand_dir.is_dir():
        return cand_dir

    # Alternate naming style
    cand2 = gt_root / f"{stem}_mask.png"
    if cand2.exists():
        return cand2

    return None


def load_mask(mask_ref: Optional[Path], image_size_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask_ref is None:
        return None

    h, w = image_size_hw

    if mask_ref.is_dir():
        mask = np.zeros((h, w), dtype=np.uint8)
        pngs = sorted([p for p in mask_ref.glob("*.png") if not p.name.startswith("._")])
        for p in pngs:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, m)
        return mask

    if mask_ref.is_file():
        m = cv2.imread(str(mask_ref), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    return None


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_nullable(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def make_summary(metric_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        key = f"{row['dataset']}/{row['category']}"
        by_class[key].append(row)

    class_summary = {}
    for key, rows in sorted(by_class.items()):
        cls_acc = [1.0 if r["pred_is_anomaly"] == r["gt_is_anomaly"] else 0.0 for r in rows]
        class_summary[key] = {
            "num_samples": len(rows),
            "classification_acc": round(float(sum(cls_acc) / len(cls_acc)), 4),
            "mean_box_iou": _mean_nullable(r.get("box_iou") for r in rows),
            "mean_bbox_mask_iou": _mean_nullable(r.get("bbox_mask_iou") for r in rows),
            "mean_bbox_mask_precision": _mean_nullable(r.get("bbox_mask_precision") for r in rows),
            "mean_bbox_mask_recall": _mean_nullable(r.get("bbox_mask_recall") for r in rows),
        }

    all_acc = [1.0 if r["pred_is_anomaly"] == r["gt_is_anomaly"] else 0.0 for r in metric_rows]
    summary = {
        "num_samples": len(metric_rows),
        "classification_acc": round(float(sum(all_acc) / len(all_acc)), 4) if all_acc else None,
        "mean_box_iou": _mean_nullable(r.get("box_iou") for r in metric_rows),
        "mean_bbox_mask_iou": _mean_nullable(r.get("bbox_mask_iou") for r in metric_rows),
        "mean_bbox_mask_precision": _mean_nullable(r.get("bbox_mask_precision") for r in metric_rows),
        "mean_bbox_mask_recall": _mean_nullable(r.get("bbox_mask_recall") for r in metric_rows),
        "by_class": class_summary,
    }
    return summary


def visualize_sample(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    pred_bbox: Optional[List[int]],
    gt_mask: Optional[np.ndarray],
    gt_bbox: Optional[List[int]],
    *,
    sample_title: str,
    pred_score: float,
    pred_is_anomaly: bool,
    gt_is_anomaly: bool,
    metrics: Dict[str, Optional[float]],
    panel_size: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    heatmap = make_heatmap(anomaly_map, (w, h))
    highlight = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0.0)

    pred_color = (0, 0, 255)
    gt_color = (0, 255, 0)
    mask_color = (0, 140, 255)

    orig = image.copy()
    orig = draw_bbox(orig, pred_bbox, pred_color, "pred bbox")
    pred_text = f"pred={'anomaly' if pred_is_anomaly else 'normal'} score={pred_score:.4f}"
    gt_text = f"gt={'anomaly' if gt_is_anomaly else 'normal'}"
    cv2.putText(orig, pred_text, (8, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(orig, gt_text, (8, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    mask_panel = image.copy()
    if gt_mask is not None:
        mask_panel = overlay_mask(mask_panel, gt_mask, mask_color, alpha=0.45)
        mask_panel = draw_mask_contour(mask_panel, gt_mask, gt_color)
        mask_panel = draw_bbox(mask_panel, gt_bbox, gt_color, "gt bbox")
    else:
        cv2.putText(mask_panel, "no gt mask", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    heatmap_panel = heatmap
    bbox_vs_mask = image.copy()
    bbox_vs_mask = draw_bbox(bbox_vs_mask, pred_bbox, pred_color, "pred")
    bbox_vs_mask = draw_bbox(bbox_vs_mask, gt_bbox, gt_color, "gt")
    if gt_mask is not None:
        bbox_vs_mask = draw_mask_contour(bbox_vs_mask, gt_mask, (255, 255, 0))

    metric_card = np.zeros_like(image)
    lines = [
        sample_title,
        f"box_iou: {metrics['box_iou'] if metrics['box_iou'] is not None else 'n/a'}",
        f"bbox_mask_iou: {metrics['bbox_mask_iou'] if metrics['bbox_mask_iou'] is not None else 'n/a'}",
        f"bbox_mask_precision: {metrics['bbox_mask_precision'] if metrics['bbox_mask_precision'] is not None else 'n/a'}",
        f"bbox_mask_recall: {metrics['bbox_mask_recall'] if metrics['bbox_mask_recall'] is not None else 'n/a'}",
    ]
    y = 38
    for line in lines:
        cv2.putText(metric_card, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        y += 36

    p1 = add_title(to_panel(orig, panel_size), "Original + Pred BBox")
    p2 = add_title(to_panel(mask_panel, panel_size), "Mask Overlay")
    p3 = add_title(to_panel(heatmap_panel, panel_size), "Heatmap")
    p4 = add_title(to_panel(highlight, panel_size), "Highlight")
    p5 = add_title(to_panel(bbox_vs_mask, panel_size), "BBox vs Mask")
    p6 = add_title(to_panel(metric_card, panel_size), "Metrics")

    top = np.hstack([p1, p2, p3])
    bottom = np.hstack([p4, p5, p6])
    return np.vstack([top, bottom])


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore subset inference + bbox/mask audit")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="PatchCore checkpoint root")
    parser.add_argument("--data-root", type=str, default="/Volumes/T7/Dataset/MMAD", help="Dataset root")
    parser.add_argument("--mmad-json", type=str, default="/Volumes/T7/Dataset/MMAD/mmad_10classes.json", help="MMAD json path")
    parser.add_argument("--output-dir", type=str, default="output/audit_subset", help="Output root")
    parser.add_argument("--samples-per-class", type=int, default=4, help="How many test images per class")
    parser.add_argument("--include-good", action="store_true", help="Include normal(good) samples")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Dataset filter")
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="Category filter")
    parser.add_argument("--version", type=int, default=None, help="Checkpoint version override")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--input-size", type=int, nargs=2, default=[700, 700], help="Model input [H W]")
    parser.add_argument("--default-threshold", type=float, default=0.5, help="Fallback threshold")
    parser.add_argument("--allow-online-backbone", action="store_true", help="Allow online backbone fallback when needed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-size", type=int, default=420, help="Single panel image size")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_root = Path(args.data_root)
    mmad_json = Path(args.mmad_json)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint-dir not found: {checkpoint_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")
    if not mmad_json.exists():
        raise FileNotFoundError(f"mmad-json not found: {mmad_json}")

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    mmad_entries = load_mmad_entries(mmad_json)
    sampled = sample_records(
        mmad_entries,
        datasets=args.datasets,
        categories=args.categories,
        include_good=args.include_good,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    if not sampled:
        print("No samples matched filters.")
        return

    print(f"Sampled {len(sampled)} images")
    by_cls: Dict[str, int] = defaultdict(int)
    for rec in sampled:
        by_cls[f"{rec.dataset}/{rec.category}"] += 1
    for key, count in sorted(by_cls.items()):
        print(f"  - {key}: {count}")

    runner = PatchCoreCkptRunner(
        checkpoint_dir=checkpoint_dir,
        version=args.version,
        device=args.device,
        input_size=(int(args.input_size[0]), int(args.input_size[1])),
        allow_online_backbone=args.allow_online_backbone,
        default_threshold=float(args.default_threshold),
    )

    predictions: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    failures = 0

    grouped: Dict[Tuple[str, str], List[SampleRecord]] = defaultdict(list)
    for rec in sampled:
        grouped[(rec.dataset, rec.category)].append(rec)

    pbar = tqdm(total=len(sampled), desc="audit", ncols=120)

    for (dataset, category), recs in sorted(grouped.items()):
        try:
            runner.warmup(dataset, category)
        except Exception as exc:
            print(f"[{dataset}/{category}] model load failed: {exc}")
            failures += len(recs)
            pbar.update(len(recs))
            continue

        for rec in recs:
            image_path = resolve_image_path(rec, data_root)
            if image_path is None:
                failures += 1
                pbar.update(1)
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                failures += 1
                pbar.update(1)
                continue

            try:
                pred = runner.predict(dataset, category, image)
            except Exception:
                failures += 1
                pbar.update(1)
                continue

            defect_loc = compute_defect_location(pred["anomaly_map"], pred["threshold"])
            pred_bbox = defect_loc.get("bbox")
            gt_is_anomaly = rec.defect_type != "good"
            gt_mask_ref = resolve_mask_ref(rec, data_root)
            gt_mask = load_mask(gt_mask_ref, image.shape[:2])
            gt_bbox = mask_to_bbox(gt_mask) if gt_mask is not None else None

            bbox_metrics = compute_bbox_mask_metrics(pred_bbox, gt_mask)

            pred_row = {
                "image_path": rec.rel_path,
                "dataset": dataset,
                "category": category,
                "defect_type": rec.defect_type,
                "anomaly_score": round(float(pred["anomaly_score"]), 6),
                "is_anomaly": bool(pred["is_anomaly"]),
                "gt_is_anomaly": bool(gt_is_anomaly),
                "threshold": float(pred["threshold"]),
                "defect_location": defect_loc,
                "mask_path": str(gt_mask_ref) if gt_mask_ref is not None else None,
            }
            predictions.append(pred_row)

            metric_row = {
                "dataset": dataset,
                "category": category,
                "defect_type": rec.defect_type,
                "image_path": rec.rel_path,
                "anomaly_score": round(float(pred["anomaly_score"]), 6),
                "pred_is_anomaly": int(bool(pred["is_anomaly"])),
                "gt_is_anomaly": int(bool(gt_is_anomaly)),
                "pred_bbox": json.dumps(pred_bbox) if pred_bbox is not None else "",
                "gt_bbox": json.dumps(gt_bbox) if gt_bbox is not None else "",
                "box_iou": bbox_metrics["box_iou"],
                "bbox_mask_iou": bbox_metrics["bbox_mask_iou"],
                "bbox_mask_precision": bbox_metrics["bbox_mask_precision"],
                "bbox_mask_recall": bbox_metrics["bbox_mask_recall"],
            }
            metric_rows.append(metric_row)

            vis = visualize_sample(
                image=image,
                anomaly_map=pred["anomaly_map"],
                pred_bbox=pred_bbox,
                gt_mask=gt_mask,
                gt_bbox=gt_bbox,
                sample_title=f"{dataset}/{category}/{rec.defect_type}/{Path(rec.rel_path).name}",
                pred_score=float(pred["anomaly_score"]),
                pred_is_anomaly=bool(pred["is_anomaly"]),
                gt_is_anomaly=bool(gt_is_anomaly),
                metrics=bbox_metrics,
                panel_size=int(args.panel_size),
            )

            stem = Path(rec.rel_path).stem
            vis_name = f"{dataset}_{category}_{rec.defect_type}_{stem}.png".replace("/", "_")
            cv2.imwrite(str(vis_dir / vis_name), vis)
            pbar.update(1)

        runner.clear()

    pbar.close()

    summary = make_summary(metric_rows)

    pred_json_path = output_dir / "predictions_subset.json"
    metrics_csv_path = output_dir / "bbox_mask_metrics.csv"
    summary_json_path = output_dir / "bbox_mask_summary.json"
    save_json(pred_json_path, predictions)
    save_csv(metrics_csv_path, metric_rows)
    save_json(summary_json_path, summary)

    print()
    print("=" * 70)
    print("Audit complete")
    print("=" * 70)
    print(f"Processed: {len(predictions)}")
    print(f"Failures: {failures}")
    print(f"Visualizations: {vis_dir}")
    print(f"Predictions JSON: {pred_json_path}")
    print(f"BBox/Mask metrics CSV: {metrics_csv_path}")
    print(f"Summary JSON: {summary_json_path}")

    if summary.get("num_samples", 0) > 0:
        print(f"Overall classification_acc: {summary.get('classification_acc')}")
        print(f"Overall mean_box_iou: {summary.get('mean_box_iou')}")
        print(f"Overall mean_bbox_mask_iou: {summary.get('mean_bbox_mask_iou')}")


if __name__ == "__main__":
    main()

