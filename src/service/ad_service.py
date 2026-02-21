from __future__ import annotations

import inspect
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class AdService:
    """PatchCore checkpoint inference service for uploaded images."""

    def __init__(
        self,
        checkpoint_dir: str,
        output_root: str,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model_cache: Dict[str, Any] = {}

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_model(self, dataset: str, category: str):
        del dataset  # kept for API compatibility
        model_path = self.checkpoint_dir / category / "v0" / "model.ckpt"
        model_key = str(model_path)
        if model_key in self.model_cache:
            return self.model_cache[model_key]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        from anomalib.models import Patchcore

        load_kwargs: Dict[str, Any] = {
            "map_location": "cpu",
            "weights_only": False,
            "pre_trained": False,
        }

        try:
            model = Patchcore.load_from_checkpoint(str(model_path), **load_kwargs)
        except TypeError as exc:
            if "pre_trained" not in str(exc):
                raise
            load_kwargs.pop("pre_trained", None)
            model = Patchcore.load_from_checkpoint(str(model_path), **load_kwargs)
        except Exception:
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            hyper = ckpt.get("hyper_parameters", {})
            sig = inspect.signature(Patchcore.__init__)
            allowed = set(sig.parameters.keys()) - {"self"}
            init_kwargs = {k: v for k, v in hyper.items() if k in allowed}
            init_kwargs["pre_trained"] = False
            model = Patchcore(**init_kwargs)
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        model.to(self.device)
        self.model_cache[model_key] = model
        return model

    @torch.no_grad()
    def predict_batch(
        self,
        upload_files: List[Any],
        category: str,
        dataset: str,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        model = self.get_model(dataset, category)
        inputs = []
        originals: List[Image.Image] = []
        request_ids: List[str] = []

        for f in upload_files:
            req_id = str(uuid.uuid4())[:8]
            img = Image.open(f.file).convert("RGB")
            inputs.append(self.transform(img))
            originals.append(img)
            request_ids.append(req_id)

        batch = torch.stack(inputs).to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast(
            device_type="cuda" if self.device.type == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(batch)

        if hasattr(outputs, "anomaly_map") and hasattr(outputs, "pred_score"):
            batch_maps = outputs.anomaly_map
            batch_scores = outputs.pred_score
        elif isinstance(outputs, (list, tuple)):
            batch_maps, batch_scores = outputs
        else:
            batch_maps = outputs
            batch_scores = outputs.view(len(upload_files), -1).max(dim=1).values

        results: List[Dict[str, Any]] = []
        for i in range(len(upload_files)):
            amap = batch_maps[i].squeeze().cpu().numpy().astype(np.float32)
            amap = cv2.GaussianBlur(amap, (3, 3), 0)

            w_orig, h_orig = originals[i].size
            amap_resized = cv2.resize(amap, (224, 224))
            amap_final = cv2.resize(amap_resized, (w_orig, h_orig))

            orig_path = self.output_root / f"orig_{request_ids[i]}.png"
            heatmap_path = self.output_root / f"heat_{request_ids[i]}.png"
            mask_path = self.output_root / f"mask_{request_ids[i]}.png"
            overlay_path = self.output_root / f"overlay_{request_ids[i]}.png"

            originals[i].save(orig_path)
            loc = self._save_visuals(
                orig_img=originals[i],
                amap=amap_final,
                h_path=heatmap_path,
                m_path=mask_path,
                o_path=overlay_path,
                threshold=threshold,
            )

            results.append(
                {
                    "ad_score": round(float(batch_scores[i].cpu().item()), 6),
                    "original_path": str(orig_path),
                    "heatmap_path": str(heatmap_path),
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "region": loc.get("region"),
                    "has_defect": loc.get("has_defect", False),
                    "area_ratio": loc.get("area_ratio"),
                    "bbox": loc.get("bbox"),
                    "center": loc.get("center"),
                    "confidence": loc.get("confidence"),
                }
            )
        return results

    def _save_visuals(
        self,
        *,
        orig_img: Image.Image,
        amap: np.ndarray,
        h_path: Path,
        m_path: Path,
        o_path: Path,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        input_img = np.array(orig_img)
        h, w = amap.shape
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        amap_8bit = (amap_norm * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(amap_8bit, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        Image.fromarray(cv2.addWeighted(input_img, 0.5, heatmap_rgb, 0.5, 0)).save(h_path)

        mask = (amap_norm > threshold).astype(np.uint8) * 255
        loc: Dict[str, Any] = {
            "has_defect": False,
            "region": "none",
            "bbox": None,
            "center": None,
            "area_ratio": 0.0,
            "confidence": 0.0,
        }

        if np.any(mask > 0):
            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            coords = np.where(mask_cleaned > 0)
            if len(coords[0]) > 0:
                y_min, y_max = int(coords[0].min()), int(coords[0].max())
                x_min, x_max = int(coords[1].min()), int(coords[1].max())

                center_y = (y_min + y_max) / 2 / h
                center_x = (x_min + x_max) / 2 / w

                ry = "top" if center_y < 0.33 else ("bottom" if center_y > 0.66 else "middle")
                rx = "left" if center_x < 0.33 else ("right" if center_x > 0.66 else "center")
                region = "center" if (ry == "middle" and rx == "center") else (f"{ry}-{rx}" if ry != "middle" else rx)

                area_ratio = round(float(np.sum(mask_cleaned > 0)) / (h * w), 4)
                confidence = round(float(amap_norm[mask_cleaned > 0].max()), 4)
                loc.update(
                    {
                        "has_defect": True,
                        "region": region,
                        "bbox": [x_min, y_min, x_max, y_max],
                        "center": [round(center_x, 3), round(center_y, 3)],
                        "area_ratio": area_ratio,
                        "confidence": confidence,
                    }
                )

        Image.fromarray(mask).save(m_path)
        highlight = input_img.copy()
        highlight[mask > 0] = [255, 0, 0]
        mask_overlay = cv2.addWeighted(input_img, 0.7, highlight, 0.3, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_overlay, contours, -1, (255, 0, 0), 2)
        Image.fromarray(mask_overlay).save(o_path)
        return loc
