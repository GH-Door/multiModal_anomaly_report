"""PatchCore evaluator module."""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .dataset import get_dataloader
from .model import PatchCore


class PatchCoreEvaluator:
    """Evaluator for PatchCore models.

    Handles evaluation metrics computation for anomaly detection.
    """

    def __init__(
        self,
        config: Dict,
        device: torch.device = None,
    ):
        """Initialize evaluator.

        Args:
            config: Configuration dictionary
            device: Device to use
        """
        self.config = config

        if device is None:
            device_str = config.get("device", "cuda")
            if device_str == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
            elif device_str == "mps" and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        # Data settings
        self.data_root = Path(config["data"]["root"])
        self.image_size = config["data"].get("image_size", 224)

        # Evaluation settings
        eval_config = config.get("evaluation", {})
        self.batch_size = eval_config.get("batch_size", 64)  # Larger batch for speed
        self.num_workers = config.get("training", {}).get("num_workers", 8)  # More workers

        # Output settings
        output_config = config.get("output", {})
        self.checkpoint_dir = Path(output_config.get("checkpoint_dir", "checkpoints"))

    def evaluate_category(
        self,
        model: PatchCore,
        dataset_name: str,
        category: str,
        verbose: bool = True,
    ) -> Dict:
        """Evaluate model on a single category.

        Args:
            model: PatchCore model
            dataset_name: Dataset name
            category: Category name
            verbose: Print progress

        Returns:
            Dictionary with evaluation metrics (image_auroc, pixel_auroc, pro)
        """
        if verbose:
            print(f"\nEvaluating: {dataset_name}/{category}")

        model.to(self.device)
        model.eval()

        # Create test dataloader with more workers for speed
        dataloader = get_dataloader(
            root=self.data_root,
            dataset_name=dataset_name,
            category=category,
            split="test",
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            include_mask=True,
        )

        if len(dataloader.dataset) == 0:
            print(f"Warning: No test samples found for {dataset_name}/{category}")
            return {}

        # Collect predictions
        all_scores = []
        all_labels = []
        all_maps = []
        all_masks = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].numpy()
                masks = batch["mask"].numpy()

                scores, maps = model.predict(images)

                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels)
                all_maps.append(maps.cpu().numpy())
                all_masks.append(masks)

        # Concatenate results
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        all_maps = np.concatenate(all_maps)
        all_masks = np.concatenate(all_masks).squeeze(1)  # Remove channel dim

        # Compute image-level metrics
        metrics = self._compute_image_metrics(all_scores, all_labels)

        # Compute pixel-level metrics if masks available
        if all_masks.max() > 0:
            pixel_metrics = self._compute_pixel_metrics(all_maps, all_masks)
            metrics.update(pixel_metrics)

        if verbose:
            print(f"  Image AUROC: {metrics.get('image_auroc', 0):.4f}, "
                  f"Pixel AUROC: {metrics.get('pixel_auroc', 0):.4f}, "
                  f"PRO: {metrics.get('pro', 0):.4f}")

        return metrics

    def _compute_image_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        """Compute image-level metrics.

        Args:
            scores: Anomaly scores (N,)
            labels: Ground truth labels (N,)

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "image_auroc": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0,
        }
        return metrics

    def _compute_pixel_metrics(
        self,
        maps: np.ndarray,
        masks: np.ndarray,
    ) -> Dict:
        """Compute pixel-level metrics (pixel AUROC and PRO).

        Args:
            maps: Anomaly maps (N, H, W)
            masks: Ground truth masks (N, H, W)

        Returns:
            Dictionary with pixel metrics
        """
        # Flatten for pixel-level AUROC
        maps_flat = maps.flatten()
        masks_flat = (masks > 0).astype(int).flatten()

        # Only compute if there are both positive and negative pixels
        if len(np.unique(masks_flat)) < 2:
            return {}

        metrics = {
            "pixel_auroc": roc_auc_score(masks_flat, maps_flat),
            "pro": self._compute_pro(maps, masks),
        }
        return metrics

    def _compute_pro(
        self,
        maps: np.ndarray,
        masks: np.ndarray,
        fpr_limit: float = 0.3,
        num_thresholds: int = 50,
    ) -> float:
        """Compute Per-Region Overlap (PRO) score.

        Optimized: precompute connected components once, not per-threshold.

        Args:
            maps: Anomaly maps (N, H, W)
            masks: Ground truth masks (N, H, W)
            fpr_limit: FPR integration limit (default 0.3)
            num_thresholds: Number of thresholds to evaluate

        Returns:
            PRO score (0-1)
        """
        masks_binary = (masks > 0).astype(np.float32)

        # Precompute connected components for all masks (do this ONCE)
        region_info = []  # List of (image_idx, region_mask, region_size)
        for i in range(len(masks)):
            if masks_binary[i].sum() == 0:
                continue
            labeled_mask, num_regions = connected_components(masks_binary[i])
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_mask == region_id)
                region_info.append((i, region_mask, region_mask.sum()))

        if not region_info:
            return 0.0

        # Precompute total negative pixels
        total_neg = (masks_binary == 0).sum()
        if total_neg == 0:
            return 0.0

        # Get thresholds from anomaly map values
        min_val, max_val = maps.min(), maps.max()
        thresholds = np.linspace(max_val, min_val, num_thresholds)

        # Compute FPR and per-region overlap at each threshold
        fprs = np.zeros(num_thresholds)
        pros = np.zeros(num_thresholds)

        for t_idx, thresh in enumerate(thresholds):
            preds = (maps >= thresh)

            # FPR
            fp = (preds & (masks_binary == 0)).sum()
            fprs[t_idx] = fp / total_neg

            # Per-region overlap using precomputed regions
            overlaps = np.array([
                preds[img_idx][region_mask].sum() / region_size
                for img_idx, region_mask, region_size in region_info
            ])
            pros[t_idx] = overlaps.mean()

        # Sort by FPR
        sorted_idx = np.argsort(fprs)
        fprs = fprs[sorted_idx]
        pros = pros[sorted_idx]

        # Integrate up to fpr_limit
        valid_idx = fprs <= fpr_limit
        if valid_idx.sum() < 2:
            return 0.0

        # Normalize FPR to [0, 1] range within the limit
        fprs_norm = fprs[valid_idx] / fpr_limit

        # Compute area under curve using trapezoidal rule
        pro_score = np.trapz(pros[valid_idx], fprs_norm)

        return float(pro_score)

    def evaluate_all(
        self,
        models: Dict[str, PatchCore],
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Evaluate all loaded models.

        Args:
            models: Dictionary mapping "dataset/category" to models
            verbose: Print progress

        Returns:
            Dictionary mapping "dataset/category" to metrics
        """
        results = {}
        total = len(models)

        print(f"\nEvaluating {total} models")

        pbar = tqdm(models.items(), desc="Evaluating", total=total)
        for key, model in pbar:
            pbar.set_description(f"{key}")

            parts = key.split("/")
            dataset_name, category = parts[0], parts[1]

            metrics = self.evaluate_category(
                model=model,
                dataset_name=dataset_name,
                category=category,
                verbose=False,
            )

            results[key] = metrics

            # Show metrics in progress bar
            pbar.set_postfix({
                "I-AUC": f"{metrics.get('image_auroc', 0):.3f}",
                "P-AUC": f"{metrics.get('pixel_auroc', 0):.3f}",
                "PRO": f"{metrics.get('pro', 0):.3f}",
            })

        # Compute average metrics
        if results:
            avg_metrics = self._compute_average_metrics(results)
            results["average"] = avg_metrics

            print(f"\n{'='*60}")
            print("Average Metrics:")
            print(f"  Image AUROC: {avg_metrics.get('image_auroc', 0):.4f}")
            print(f"  Pixel AUROC: {avg_metrics.get('pixel_auroc', 0):.4f}")
            print(f"  PRO:         {avg_metrics.get('pro', 0):.4f}")
            print(f"{'='*60}")

        return results

    def _compute_average_metrics(self, results: Dict[str, Dict]) -> Dict:
        """Compute average metrics across all categories.

        Args:
            results: Dictionary of per-category metrics

        Returns:
            Dictionary with averaged metrics
        """
        metric_keys = ["image_auroc", "pixel_auroc", "pro"]
        avg_metrics = {}

        for metric_key in metric_keys:
            values = [
                metrics[metric_key]
                for key, metrics in results.items()
                if key != "average" and metric_key in metrics
            ]
            if values:
                avg_metrics[metric_key] = np.mean(values)

        return avg_metrics

    def save_results(self, results: Dict[str, Dict], output_path: str) -> None:
        """Save evaluation results to JSON.

        Args:
            results: Evaluation results dictionary
            output_path: Output file path
        """
        import json

        # Convert numpy types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj

        results_serializable = convert_to_serializable(results)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")
