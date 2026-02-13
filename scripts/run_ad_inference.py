"""Unified PatchCore inference runner (checkpoint + ONNX backends).

This script generates AD prediction JSON for downstream LLM evaluation.

Examples:
    # Checkpoint backend (recommended for now)
    python scripts/run_ad_inference.py \
        --backend ckpt \
        --checkpoint-dir /path/to/checkpoints \
        --data-root /path/to/MMAD \
        --mmad-json /path/to/mmad.json \
        --output output/ad_predictions.json \
        --device cuda

    # ONNX backend (future deployment)
    python scripts/run_ad_inference.py \
        --backend onnx \
        --models-dir models/onnx \
        --data-root /path/to/MMAD \
        --mmad-json /path/to/mmad.json \
        --output output/ad_predictions_onnx.json
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.anomaly import PatchCoreModelManager
from src.utils.loaders import load_config


LOW_RELIABILITY_CLASSES = {
    "screw_bag",
    "pushpins",
    "breakfast_box",
    "juice_bottle",
}


def parse_image_path(image_path: str) -> Tuple[str, str]:
    parts = image_path.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def _is_hub_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keys = [
        "locate the file on the hub",
        "hf_hub_offline",
        "huggingface.co",
        "local cache",
    ]
    return any(k in msg for k in keys)


def compute_confidence_level(anomaly_score: float, category: str) -> Dict[str, Any]:
    is_low_reliability = category in LOW_RELIABILITY_CLASSES

    if anomaly_score > 0.8:
        level = "high"
        reason = "Strong anomaly signal (score > 0.8)"
    elif anomaly_score < 0.2:
        level = "high"
        reason = "Strong normal signal (score < 0.2)"
    elif anomaly_score > 0.6 or anomaly_score < 0.4:
        level = "medium"
        reason = "Moderate confidence (score between 0.4-0.6 boundary)"
    else:
        level = "low"
        reason = "Score near decision boundary (0.4-0.6)"

    if is_low_reliability:
        reliability = "low"
        reliability_reason = (
            f"Class '{category}' contains logical anomalies that PatchCore cannot reliably detect"
        )
    else:
        reliability = "high"
        reliability_reason = "Structural anomaly class - PatchCore reliable"

    return {
        "level": level,
        "reliability": reliability,
        "reason": reason,
        "reliability_reason": reliability_reason,
    }


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

    area_ratio = float(defect_mask.sum()) / (h * w)
    confidence = float(anomaly_map[defect_mask].max())

    return {
        "has_defect": True,
        "region": region,
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "center": [round(center_x, 3), round(center_y, 3)],
        "area_ratio": round(area_ratio, 4),
        "confidence": round(confidence, 4),
    }


class PatchCoreCheckpointRunner:
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        version: Optional[int],
        threshold: float,
        device: str,
        input_size: Tuple[int, int],
        allow_online_backbone: bool,
        sync_timing: bool,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.version = version
        self.threshold = threshold
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.input_size = input_size
        self.allow_online_backbone = allow_online_backbone
        self.sync_timing = sync_timing

        self._models: Dict[str, Any] = {}
        self._warmup_done: set[str] = set()

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

        versions = []
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

    def list_available_models(self) -> List[Tuple[str, str]]:
        available = []
        patchcore_dir = self.checkpoint_dir / "Patchcore"
        if not patchcore_dir.exists():
            patchcore_dir = self.checkpoint_dir

        if not patchcore_dir.exists():
            return available

        for dataset_dir in patchcore_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            if dataset_dir.name in ["eval", "predictions"]:
                continue

            for category_dir in dataset_dir.iterdir():
                if not category_dir.is_dir() or category_dir.name.startswith("."):
                    continue

                if self._find_checkpoint(dataset_dir.name, category_dir.name) is not None:
                    available.append((dataset_dir.name, category_dir.name))

        return sorted(available)

    @staticmethod
    def _load_from_checkpoint(ckpt_path: Path, *, pre_trained_override: Optional[bool]):
        from anomalib.models import Patchcore

        load_kwargs: Dict[str, Any] = {
            "map_location": "cpu",
            "weights_only": False,
        }
        if pre_trained_override is not None:
            load_kwargs["pre_trained"] = pre_trained_override

        try:
            return Patchcore.load_from_checkpoint(str(ckpt_path), **load_kwargs)
        except TypeError as exc:
            if "pre_trained" in str(exc):
                load_kwargs.pop("pre_trained", None)
                return Patchcore.load_from_checkpoint(str(ckpt_path), **load_kwargs)
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
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: missing keys during manual checkpoint load ({len(missing)})")
        if unexpected:
            print(f"  Warning: unexpected keys during manual checkpoint load ({len(unexpected)})")
        return model

    def get_model(self, dataset: str, category: str):
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

            print("  Warning: Hub lookup error during checkpoint load; retrying with no-download loader")
            try:
                model = self._load_manual_no_hub(ckpt_path)
            except Exception as manual_exc:
                if self.allow_online_backbone:
                    print("  Warning: no-download loader failed; retrying with online backbone resolution")
                    model = self._load_from_checkpoint(ckpt_path, pre_trained_override=None)
                else:
                    raise RuntimeError(
                        "Checkpoint load failed due to Hub resolution. "
                        "Retry with --allow-online-backbone to permit online download."
                    ) from manual_exc

        model.eval()
        model.to(self.device)
        self._models[key] = model
        return model

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        h, w = self.input_size
        img = cv2.resize(image, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return tensor.to(self.device)

    def warmup_model(self, dataset: str, category: str) -> None:
        key = f"{dataset}/{category}"
        if key in self._warmup_done:
            return

        model = self.get_model(dataset, category)
        dummy = torch.randn(1, 3, self.input_size[0], self.input_size[1], device=self.device)
        with torch.inference_mode():
            _ = model(dummy)
        if self.sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize()
        self._warmup_done.add(key)

    def predict(self, dataset: str, category: str, image: np.ndarray) -> Dict[str, Any]:
        model = self.get_model(dataset, category)
        input_tensor = self._preprocess(image)
        original_size = image.shape[:2]

        with torch.inference_mode():
            outputs = model(input_tensor)
            if self.sync_timing and self.device.type == "cuda":
                torch.cuda.synchronize()

        anomaly_map = getattr(outputs, "anomaly_map", None)
        pred_score = getattr(outputs, "pred_score", None)
        pred_label = getattr(outputs, "pred_label", None)

        if anomaly_map is not None:
            if anomaly_map.shape[-2:] != (original_size[0], original_size[1]):
                anomaly_map = interpolate(anomaly_map, size=original_size, mode="bilinear", align_corners=False)
            anomaly_map = anomaly_map[0, 0].detach().cpu().numpy()

        anomaly_score = float(pred_score[0].detach().cpu()) if pred_score is not None else float(np.max(anomaly_map))
        is_anomaly = bool(pred_label[0].detach().cpu()) if pred_label is not None else (anomaly_score > self.threshold)

        threshold_used = self.threshold
        post = getattr(model, "post_processor", None)
        if post is not None and hasattr(post, "normalized_image_threshold"):
            thr = post.normalized_image_threshold
            if isinstance(thr, torch.Tensor):
                threshold_used = float(thr.detach().cpu())
            else:
                threshold_used = float(thr)

        del input_tensor, outputs
        if pred_score is not None:
            del pred_score
        if pred_label is not None:
            del pred_label

        return {
            "anomaly_score": anomaly_score,
            "anomaly_map": anomaly_map,
            "is_anomaly": is_anomaly,
            "threshold": threshold_used,
            "backend": "ckpt",
        }

    def clear_cache(self) -> None:
        self._models.clear()
        self._warmup_done.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class PatchCoreOnnxRunner:
    def __init__(self, models_dir: Path, *, threshold: float, device: str):
        self.manager = PatchCoreModelManager(models_dir=models_dir, threshold=threshold, device=device)

    def list_available_models(self) -> List[Tuple[str, str]]:
        return self.manager.list_available_models()

    def warmup_model(self, dataset: str, category: str) -> None:
        # ONNXRuntime session is initialized lazily on first get_model.
        _ = self.manager.get_model(dataset, category)

    def predict(self, dataset: str, category: str, image: np.ndarray) -> Dict[str, Any]:
        result = self.manager.predict(dataset, category, image)
        return {
            "anomaly_score": float(result.anomaly_score),
            "anomaly_map": result.anomaly_map,
            "is_anomaly": bool(result.is_anomaly),
            "threshold": float(result.threshold),
            "backend": "onnx",
        }

    def clear_cache(self) -> None:
        self.manager.clear_cache()


def load_images(mmad_json: Path, *, datasets: Optional[List[str]], categories: Optional[List[str]]) -> List[str]:
    with open(mmad_json, "r", encoding="utf-8") as f:
        mmad_data = json.load(f)

    image_paths = list(mmad_data.keys())

    if datasets or categories:
        filtered = []
        datasets_set = set(datasets or [])
        categories_set = set(categories or [])
        for path in image_paths:
            dataset, category = parse_image_path(path)
            if datasets_set and dataset not in datasets_set:
                continue
            if categories_set and category not in categories_set:
                continue
            filtered.append(path)
        image_paths = filtered

    return image_paths


def save_results(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def build_result_dict(
    image_path: str,
    category: str,
    pred: Dict[str, Any],
    *,
    include_map_stats: bool,
) -> Dict[str, Any]:
    anomaly_score = round(float(pred["anomaly_score"]), 4)
    threshold = float(pred.get("threshold", 0.5))
    anomaly_map = pred.get("anomaly_map")
    is_anomaly = bool(pred["is_anomaly"])

    out: Dict[str, Any] = {
        "image_path": image_path,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "threshold": threshold,
        "backend": pred.get("backend"),
        "confidence": compute_confidence_level(anomaly_score, category),
    }

    if anomaly_map is not None:
        if is_anomaly:
            out["defect_location"] = compute_defect_location(anomaly_map, threshold)
        else:
            out["defect_location"] = {
                "has_defect": False,
                "region": "none",
                "bbox": None,
                "center": None,
                "area_ratio": 0.0,
            }

        if include_map_stats:
            out["map_stats"] = {
                "max": round(float(np.max(anomaly_map)), 4),
                "mean": round(float(np.mean(anomaly_map)), 4),
                "std": round(float(np.std(anomaly_map)), 4),
            }

    return out


def resolve_config(args: argparse.Namespace) -> Tuple[Optional[int], Optional[List[str]], Optional[List[str]], Tuple[int, int]]:
    config_version = args.version
    config_datasets = args.datasets
    config_categories = args.categories
    input_size = tuple(args.input_size) if args.input_size is not None else None

    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        if config_version is None:
            config_version = config.get("predict", {}).get("version")
        if config_datasets is None:
            config_datasets = config.get("data", {}).get("datasets")
        if config_categories is None:
            config_categories = config.get("data", {}).get("categories")
        if input_size is None:
            input_size = tuple(config.get("data", {}).get("image_size", [700, 700]))

    if input_size is None:
        input_size = (700, 700)

    return config_version, config_datasets, config_categories, input_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified PatchCore AD inference")

    # Backend
    parser.add_argument("--backend", type=str, default="ckpt", choices=["ckpt", "onnx"], help="Inference backend")

    # Backend-specific paths
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint root (for ckpt backend)")
    parser.add_argument("--models-dir", type=str, default="models/onnx", help="ONNX model root (for onnx backend)")

    # Common runtime
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml", help="Config file for filters/version")
    parser.add_argument("--version", type=int, default=None, help="Checkpoint version override (ckpt backend)")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Dataset filter override")
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="Category filter override")
    parser.add_argument("--threshold", type=float, default=0.5, help="Default anomaly threshold")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--input-size", type=int, nargs=2, default=None, help="Input size override [H W] for ckpt")

    # Data IO
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--mmad-json", type=str, required=True, help="Path to mmad.json")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to process")
    parser.add_argument("--output", type=str, default="output/ad_predictions.json", help="Output JSON path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--save-interval", type=int, default=200, help="Periodic save interval")
    parser.add_argument("--no-map-stats", action="store_true", help="Disable anomaly map summary stats")

    # Stability/perf knobs
    parser.add_argument("--allow-online-backbone", action="store_true", help="Allow online backbone resolution when checkpoint loading fails")
    parser.add_argument("--sync-timing", action="store_true", help="Synchronize CUDA per image (accurate timing, slower)")
    parser.add_argument("--gc-interval", type=int, default=0, help="Run gc.collect every N images (0=disabled)")
    parser.add_argument("--keep-model-cache", action="store_true", help="Keep loaded model cache across categories")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    mmad_json = Path(args.mmad_json)
    output_path = Path(args.output)
    include_map_stats = not args.no_map_stats

    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)
    if not mmad_json.exists():
        print(f"Error: MMAD JSON not found: {mmad_json}")
        sys.exit(1)

    config_version, config_datasets, config_categories, input_size = resolve_config(args)

    print(f"Backend: {args.backend}")
    print(f"Config: {args.config}")
    print(f"Version: v{config_version}" if config_version is not None else "Version: latest")
    print(f"Filters: datasets={config_datasets} categories={config_categories}")
    if args.backend == "ckpt":
        print(f"Input size: {input_size}")
    if args.backend == "ckpt":
        if not args.checkpoint_dir:
            print("Error: --checkpoint-dir is required for ckpt backend")
            sys.exit(1)
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            sys.exit(1)
        runner = PatchCoreCheckpointRunner(
            checkpoint_dir=checkpoint_dir,
            version=config_version,
            threshold=args.threshold,
            device=args.device,
            input_size=input_size,
            allow_online_backbone=args.allow_online_backbone,
            sync_timing=args.sync_timing,
        )
    else:
        models_dir = Path(args.models_dir)
        if not models_dir.exists():
            print(f"Error: Models directory not found: {models_dir}")
            sys.exit(1)
        runner = PatchCoreOnnxRunner(models_dir=models_dir, threshold=args.threshold, device=args.device)

    runtime_device = str(runner.device) if hasattr(runner, "device") else args.device
    print(f"Device: {runtime_device}")
    print()

    print(f"Loading MMAD data from: {mmad_json}")
    image_paths = load_images(mmad_json, datasets=config_datasets, categories=config_categories)
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    print(f"Total images to process: {len(image_paths)}")

    existing_results: Dict[str, Dict[str, Any]] = {}
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_list = json.load(f)
        existing_results = {r["image_path"]: r for r in existing_list}
        print(f"Loaded {len(existing_results)} existing results")

    available_models = set(runner.list_available_models())
    print(f"Available models: {len(available_models)}")
    for dataset, category in sorted(available_models):
        print(f"  - {dataset}/{category}")

    images_by_category: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    skipped_no_model = 0
    for image_path in image_paths:
        if image_path in existing_results:
            continue
        dataset, category = parse_image_path(image_path)
        if (dataset, category) not in available_models:
            skipped_no_model += 1
            continue
        images_by_category[(dataset, category)].append(image_path)

    total_to_process = sum(len(v) for v in images_by_category.values())
    print()
    print("=" * 60)
    print("Running inference")
    print("=" * 60)
    print(f"Images to process: {total_to_process} across {len(images_by_category)} categories")

    results: List[Dict[str, Any]] = list(existing_results.values())
    processed = len(existing_results)
    errors = 0
    total_inference_time = 0.0

    for (dataset, category), cat_images in images_by_category.items():
        cat_key = f"{dataset}/{category}"
        print(f"\n[{cat_key}] Loading model and processing {len(cat_images)} images...")

        try:
            runner.warmup_model(dataset, category)
        except Exception as exc:
            print(f"  Failed to load model: {exc}")
            errors += len(cat_images)
            continue

        cat_start = time.perf_counter()
        pbar = tqdm(cat_images, desc=f"  {cat_key}", ncols=100, leave=False)

        for image_path in pbar:
            image_full_path = data_root / image_path
            if not image_full_path.exists():
                errors += 1
                continue

            try:
                image = cv2.imread(str(image_full_path))
                if image is None:
                    errors += 1
                    continue

                t0 = time.perf_counter()
                pred = runner.predict(dataset, category, image)
                infer_time = time.perf_counter() - t0
                total_inference_time += infer_time

                result_dict = build_result_dict(
                    image_path,
                    category,
                    pred,
                    include_map_stats=include_map_stats,
                )
                results.append(result_dict)
                processed += 1

                if args.gc_interval > 0 and processed % args.gc_interval == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if args.save_interval > 0 and processed % args.save_interval == 0:
                    save_results(output_path, results)

            except Exception:
                errors += 1

            pbar.set_postfix({"done": processed, "err": errors})

        pbar.close()

        cat_elapsed = time.perf_counter() - cat_start
        cat_ms_per_img = (cat_elapsed / len(cat_images) * 1000.0) if cat_images else 0.0
        print(f"  Done: {len(cat_images)} images in {cat_elapsed:.1f}s ({cat_ms_per_img:.0f}ms/img)")

        if not args.keep_model_cache:
            runner.clear_cache()

    save_results(output_path, results)

    processed_from_this_run = max(0, processed - len(existing_results))
    ms_per_img = (total_inference_time / processed_from_this_run * 1000.0) if processed_from_this_run > 0 else 0.0

    print()
    print("=" * 60)
    print("Inference complete")
    print("=" * 60)
    print(f"Processed (this run): {processed_from_this_run}")
    print(f"Skipped (no model): {skipped_no_model}")
    print(f"Errors: {errors}")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Average: {ms_per_img:.1f}ms/img")
    print(f"Output saved to: {output_path}")

    if results:
        print()
        print("Sample output:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
