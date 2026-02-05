"""PatchCore model implementation.

Reference: https://arxiv.org/abs/2106.08265
"Towards Total Recall in Industrial Anomaly Detection"
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm


class FeatureExtractor(nn.Module):
    """Feature extractor using pretrained backbone.

    Extracts features from specified intermediate layers.
    """

    BACKBONES = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "wide_resnet50_2": models.wide_resnet50_2,
    }

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] = None,
        pretrained: bool = True,
    ):
        super().__init__()

        if layers is None:
            layers = ["layer2", "layer3"]

        self.layers = layers
        self.features = {}

        # Load backbone
        if backbone not in self.BACKBONES:
            raise ValueError(f"Backbone {backbone} not supported. Choose from {list(self.BACKBONES.keys())}")

        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = self.BACKBONES[backbone](weights=weights)

        # Remove classification head
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()

        # Register hooks for intermediate layers
        for layer_name in layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(self._get_hook(layer_name))

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

    def _get_hook(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from specified layers.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dictionary of layer_name -> features
        """
        self.features = {}
        _ = self.backbone(x)
        return {k: v for k, v in self.features.items()}


class PatchCore(nn.Module):
    """PatchCore anomaly detection model.

    Args:
        backbone: Backbone network name
        layers: Layers to extract features from
        coreset_ratio: Ratio of patches to keep in coreset (0-1)
        n_neighbors: Number of neighbors for anomaly scoring
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] = None,
        coreset_ratio: float = 0.01,
        n_neighbors: int = 9,
    ):
        super().__init__()

        if layers is None:
            layers = ["layer2", "layer3"]

        self.backbone_name = backbone
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors

        # Feature extractor
        self.feature_extractor = FeatureExtractor(backbone, layers)

        # Memory bank (registered as buffer so it's saved with model)
        self.register_buffer("memory_bank", torch.tensor([]))
        self.is_fitted = False

        # For tracking feature dimensions
        self._feature_dim = None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and aggregate patch features.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Patch features (B, N_patches, D)
        """
        # Get features from layers
        features_dict = self.feature_extractor(x)

        # Get features and resize to same spatial size
        features_list = []
        target_size = None

        for layer_name in self.layers:
            feat = features_dict[layer_name]
            if target_size is None:
                target_size = feat.shape[-2:]
            else:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            features_list.append(feat)

        # Concatenate along channel dimension
        features = torch.cat(features_list, dim=1)  # (B, C_total, H, W)

        # Reshape to (B, H*W, C)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)

        self._feature_dim = C
        self._spatial_size = (H, W)

        return features

    def fit(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> None:
        """Fit the model by building memory bank from training data.

        Args:
            dataloader: Training dataloader (normal samples only)
            device: Device to use
        """
        self.to(device)
        self.eval()

        all_features = []

        print("Extracting features from training data...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Feature extraction"):
                images = batch["image"].to(device)
                features = self.extract_features(images)  # (B, N, D)

                # Flatten batch and patch dimensions
                features = features.reshape(-1, features.shape[-1])  # (B*N, D)
                all_features.append(features.cpu())

        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)  # (Total_patches, D)
        print(f"Total patches extracted: {all_features.shape[0]}")

        # Apply coreset subsampling
        if self.coreset_ratio < 1.0:
            n_select = max(1, int(all_features.shape[0] * self.coreset_ratio))
            print(f"Applying coreset sampling: {all_features.shape[0]} -> {n_select}")
            indices = self._coreset_sampling(all_features, n_select)
            all_features = all_features[indices]

        # Store in memory bank
        self.memory_bank = all_features.to(device)
        self.is_fitted = True
        print(f"Memory bank size: {self.memory_bank.shape}")

    def _coreset_sampling(self, features: torch.Tensor, n_select: int) -> np.ndarray:
        """Fast coreset sampling using random selection.

        For faster training, uses random sampling instead of greedy farthest point.
        Can be changed to 'greedy' method for better quality but slower speed.

        Args:
            features: All patch features (N, D)
            n_select: Number of patches to select

        Returns:
            Indices of selected patches
        """
        N = features.shape[0]

        if n_select >= N:
            return np.arange(N)

        # Fast random sampling (recommended for large datasets)
        indices = np.random.choice(N, size=n_select, replace=False)
        return indices

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict anomaly scores and maps.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            anomaly_scores: Per-image anomaly scores (B,)
            anomaly_maps: Per-pixel anomaly maps (B, H, W)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Extract features
        features = self.extract_features(x)  # (B, N, D)
        B, N, D = features.shape

        # Compute distances to memory bank
        # features: (B, N, D), memory_bank: (M, D)
        # distances: (B, N, M)
        features_flat = features.reshape(-1, D)  # (B*N, D)

        # Compute pairwise distances
        distances = torch.cdist(features_flat, self.memory_bank)  # (B*N, M)

        # Get k nearest neighbor distances
        knn_distances, _ = distances.topk(self.n_neighbors, dim=1, largest=False)

        # Use mean of k nearest distances as patch score
        patch_scores = knn_distances.mean(dim=1)  # (B*N,)
        patch_scores = patch_scores.reshape(B, N)  # (B, N)

        # Reshape to spatial map
        H, W = self._spatial_size
        anomaly_maps = patch_scores.reshape(B, H, W)

        # Upsample to input size
        input_size = x.shape[-2:]
        anomaly_maps = F.interpolate(
            anomaly_maps.unsqueeze(1),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, H_in, W_in)

        # Image-level score: max of patch scores
        anomaly_scores = anomaly_maps.reshape(B, -1).max(dim=1)[0]  # (B,)

        return anomaly_scores, anomaly_maps

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for ONNX export.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Dictionary with anomaly_score and anomaly_map
        """
        scores, maps = self.predict(x)
        return {
            "anomaly_score": scores,
            "anomaly_map": maps,
        }

    def save(
        self,
        save_dir: str,
        dataset_name: str,
        category: str,
        save_pt: bool = True,
        save_onnx: bool = True,
        image_size: int = 224,
    ) -> Dict[str, str]:
        """Save model to disk.

        Args:
            save_dir: Base directory for saving
            dataset_name: Dataset name
            category: Category name
            save_pt: Save PyTorch checkpoint
            save_onnx: Save ONNX model
            image_size: Input image size for ONNX

        Returns:
            Dictionary of saved file paths
        """
        from pathlib import Path

        save_path = Path(save_dir) / dataset_name / category
        save_path.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        if save_pt:
            pt_path = save_path / "model.pt"
            torch.save({
                "backbone": self.backbone_name,
                "layers": self.layers,
                "coreset_ratio": self.coreset_ratio,
                "n_neighbors": self.n_neighbors,
                "memory_bank": self.memory_bank.cpu(),
                "feature_dim": self._feature_dim,
                "spatial_size": self._spatial_size,
                "is_fitted": self.is_fitted,
            }, pt_path)
            saved_paths["pt"] = str(pt_path)
            print(f"Saved PyTorch model: {pt_path}")

        if save_onnx:
            onnx_path = save_path / "model.onnx"
            self._export_onnx(onnx_path, image_size)
            saved_paths["onnx"] = str(onnx_path)
            print(f"Saved ONNX model: {onnx_path}")

        return saved_paths

    def _export_onnx(self, path: str, image_size: int = 224) -> None:
        """Export model to ONNX format."""
        # Create wrapper for ONNX export (simpler forward)
        class OnnxWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                scores, maps = self.model.predict(x)
                return scores, maps

        wrapper = OnnxWrapper(self)
        wrapper.eval()

        # Create dummy input
        device = self.memory_bank.device
        dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

        # Export
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(path),
            opset_version=14,
            input_names=["input"],
            output_names=["anomaly_score", "anomaly_map"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "anomaly_score": {0: "batch_size"},
                "anomaly_map": {0: "batch_size"},
            },
        )

    @classmethod
    def load(cls, checkpoint_path: str, device: torch.device = None) -> "PatchCore":
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to .pt file
            device: Device to load to

        Returns:
            Loaded PatchCore model
        """
        if device is None:
            device = torch.device("cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = cls(
            backbone=checkpoint["backbone"],
            layers=checkpoint["layers"],
            coreset_ratio=checkpoint["coreset_ratio"],
            n_neighbors=checkpoint["n_neighbors"],
        )

        model.memory_bank = checkpoint["memory_bank"].to(device)
        model._feature_dim = checkpoint["feature_dim"]
        model._spatial_size = checkpoint["spatial_size"]
        model.is_fitted = checkpoint["is_fitted"]

        return model.to(device)
