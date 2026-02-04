import os
os.environ["TQDM_DISABLE"] = "1"

# tqdm 강제 비활성화 (클래스 상속 유지하면서 disable=True 강제)
import tqdm
from tqdm import tqdm as tqdm_class

_original_tqdm_init = tqdm_class.__init__

def _patched_tqdm_init(self, *args, **kwargs):
    kwargs["disable"] = True
    _original_tqdm_init(self, *args, **kwargs)

tqdm_class.__init__ = _patched_tqdm_init
tqdm.tqdm = tqdm_class

import json
import time
from pathlib import Path
from anomalib.models import Patchcore, WinClip, EfficientAd
from anomalib.engine import Engine
from pytorch_lightning.callbacks import Callback

from src.utils.loaders import load_config
from src.utils.log import setup_logger
from src.utils.device import get_device
from src.datasets.dataloader import MMADLoader
logger = setup_logger(name="TrainAnomalib", log_prefix="train_anomalib")

class EpochProgressCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        metrics = trainer.callback_metrics

        parts = [f"[Epoch {epoch}/{max_epochs}]"]

        # Loss
        train_loss = metrics.get("train_loss") or metrics.get("loss")
        if train_loss is not None:
            parts.append(f"loss={float(train_loss):.4f}")

        # AUROC
        auroc = metrics.get("image_AUROC") or metrics.get("AUROC")
        if auroc is not None:
            parts.append(f"AUROC={float(auroc):.4f}")

        # F1
        f1 = metrics.get("image_F1Score") or metrics.get("F1Score")
        if f1 is not None:
            parts.append(f"F1={float(f1):.4f}")

        print(" | ".join(parts), flush=True)


class Anomalibs:
    def __init__(self, config_path: str = "configs/runtime.yaml"):
        self.config = load_config(config_path)

        # model
        self.model_name = self.config["anomaly"]["model"]
        self.model_params = self.filter_none(
            self.config["anomaly"].get(self.model_name, {})
        )

        # training
        self.training_config = self.filter_none(
            self.config.get("training", {})
        )

        # data
        self.data_root = Path(self.config["data"]["root"])
        self.output_root = Path(self.config["data"]["output_root"])

        # engine
        self.output_config = self.config.get("output", {})
        self.engine_config = self.config.get("engine", {})

        # device (for logging)
        self.device = get_device()

        # MMAD loader
        self.loader = MMADLoader(root=str(self.data_root))

        logger.info(f"Initialized - model: {self.model_name}, device: {self.device}")

    @staticmethod
    def filter_none(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    def get_model(self):
        if self.model_name == "patchcore":
            return Patchcore(**self.model_params)
        elif self.model_name == "winclip":
            return WinClip(**self.model_params)
        elif self.model_name == "efficientad":
            return EfficientAd(**self.model_params)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def get_datamodule_kwargs(self):
        # datamodule kwargs from training config
        kwargs = {}
        if "train_batch_size" in self.training_config:
            kwargs["train_batch_size"] = self.training_config["train_batch_size"]
        elif self.model_name == "efficientad":
            kwargs["train_batch_size"] = 1  # EfficientAd 1 필수
        if "eval_batch_size" in self.training_config:
            kwargs["eval_batch_size"] = self.training_config["eval_batch_size"]
        if "num_workers" in self.training_config:
            kwargs["num_workers"] = self.training_config["num_workers"]
        return kwargs

    def get_engine(self, dataset: str = None, category: str = None, model=None, datamodule=None):
        # WandB logger 설정
        logger_config = self.engine_config.get("logger", False)
        if logger_config == "wandb" and dataset and category:
            from pytorch_lightning.loggers import WandbLogger
            from src.utils.wandbs import login_wandb
            login_wandb()
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

            # batch_size 추출
            batch_size = self.training_config.get("train_batch_size")
            if batch_size is None and datamodule is not None:
                batch_size = getattr(datamodule, "train_batch_size", None)
            if batch_size is None:
                batch_size = 1 if self.model_name == "efficientad" else 32

            # max_epochs 추출
            max_epochs = self.training_config.get("max_epochs") or 100

            # model hyperparams
            lr = getattr(model, "lr", None) if model else None
            weight_decay = getattr(model, "weight_decay", None) if model else None

            logger_config = WandbLogger(
                project=self.config.get("wandb", {}).get("project", "mmad-anomaly"),
                name=f"{dataset}-{category}",
                tags=[self.model_name, dataset, category],
                config={
                    "model": self.model_name,
                    "dataset": dataset,
                    "category": category,
                    "device": gpu_name,
                    "batch_size": batch_size,
                    "epoch": max_epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
            )

        enable_progress = self.engine_config.get("enable_progress_bar", False)
        callbacks = [] if enable_progress else [EpochProgressCallback()]

        kwargs = {
            "accelerator": self.engine_config.get("accelerator", "auto"),
            "devices": 1,
            "default_root_dir": str(self.output_root),
            "logger": logger_config,
            "enable_progress_bar": enable_progress,
            "callbacks": callbacks,
        }

        if "max_epochs" in self.training_config:
            kwargs["max_epochs"] = self.training_config["max_epochs"]
        return Engine(**kwargs)

    def get_ckpt_path(self, dataset: str, category: str) -> Path | None:
        if self.model_name == "winclip":
            return None
        return (
            self.output_root
            / self.model_name.capitalize()
            / dataset
            / category
            / "v0/weights/lightning/model.ckpt"
        )

    def requires_fit(self) -> bool:
        return self.model_name != "winclip"

    def fit(self, dataset: str, category: str):
        if not self.requires_fit():
            logger.info(f"{self.model_name} - no training required (zero-shot)")
            return self

        logger.info(f"Fitting {self.model_name} - {dataset}/{category}")

        model = self.get_model()
        dm_kwargs = self.get_datamodule_kwargs()
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        engine = self.get_engine(dataset, category, model=model, datamodule=datamodule)

        engine.fit(datamodule=datamodule, model=model)

        # WandB run 종료 (카테고리별로 별도 run)
        import wandb
        if wandb.run is not None:
            wandb.finish()

        logger.info(f"Fitting {dataset}/{category} done")

        return self

    def predict(self, dataset: str, category: str, save_json: bool = None):
        logger.info(f"Predicting {self.model_name} - {dataset}/{category}")

        model = self.get_model()
        dm_kwargs = self.get_datamodule_kwargs()
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        engine = self.get_engine()
        ckpt_path = self.get_ckpt_path(dataset, category)

        # WinCLIP requires class name for text embeddings
        if self.model_name == "winclip":
            model.setup(class_name=category)

        predictions = engine.predict(
            datamodule=datamodule,
            model=model,
            ckpt_path=ckpt_path,
        )

        # save json
        if save_json is None:
            save_json = self.output_config.get("save_json", False)
        if save_json:
            self.save_predictions_json(predictions, dataset, category)

        logger.info(f"Predicting {dataset}/{category} done - {len(predictions)} batches")
        return predictions

    def get_mask_path(self, image_path: str, dataset: str) -> str | None:
        """이미지 경로에서 대응하는 마스크 경로 추론"""
        image_path = Path(image_path)

        # GoodsAD: test/{defect_type}/xxx.jpg -> ground_truth/{defect_type}/xxx.png
        if dataset == "GoodsAD":
            parts = image_path.parts
            if "test" in parts:
                test_idx = parts.index("test")
                defect_type = parts[test_idx + 1]
                # good 폴더는 마스크 없음
                if defect_type == "good":
                    return None
                mask_path = (
                    image_path.parent.parent.parent
                    / "ground_truth"
                    / defect_type
                    / (image_path.stem + ".png")
                )
                if mask_path.exists():
                    return str(mask_path)
        # MVTec-AD, VisA, MVTec-LOCO: batch에 mask_path가 이미 있음
        return None

    def save_predictions_json(self, predictions, dataset: str, category: str):
        output_dir = self.output_root / "predictions" / self.model_name / dataset / category
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for batch in predictions:
            for i in range(len(batch["image_path"])):
                image_path = str(batch["image_path"][i])
                result = {
                    "image_path": image_path,
                    "pred_score": float(batch["pred_score"][i]),
                    "pred_label": int(batch["pred_label"][i]),
                }

                # 마스크 경로 추가 (batch에 있으면 사용, 없으면 추론)
                if "mask_path" in batch and batch["mask_path"][i]:
                    result["mask_path"] = str(batch["mask_path"][i])
                else:
                    mask_path = self.get_mask_path(image_path, dataset)
                    if mask_path:
                        result["mask_path"] = mask_path

                # ground truth label (정상/비정상)
                if "label" in batch:
                    result["gt_label"] = int(batch["label"][i])

                if "anomaly_map" in batch and batch["anomaly_map"] is not None:
                    amap = batch["anomaly_map"][i]
                    result["anomaly_map_shape"] = list(amap.shape)
                    result["anomaly_map_max"] = float(amap.max())
                    result["anomaly_map_mean"] = float(amap.mean())

                results.append(result)

        json_path = output_dir / "predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved predictions JSON: {json_path}")

    def get_all_categories(self) -> list[tuple[str, str]]:
        """Get list of (dataset, category) tuples."""
        return [
            (dataset, category)
            for dataset in self.loader.DATASETS
            for category in self.loader.get_categories(dataset)
        ]

    def fit_all(self):
        categories = self.get_all_categories()
        total = len(categories)
        logger.info(f"Starting fit_all: {total} categories")

        for idx, (dataset, category) in enumerate(categories, 1):
            print(f"\n[{idx}/{total}] Training: {category}...")
            start = time.time()
            self.fit(dataset, category)
            elapsed = time.time() - start
            print(f"✓ [{idx}/{total}] {category} 완료 ({elapsed:.1f}s)")

        logger.info(f"fit_all completed: {total} categories")

    def predict_all(self, save_json: bool = None):
        categories = self.get_all_categories()
        total = len(categories)
        logger.info(f"Starting predict_all: {total} categories")

        all_predictions = {}
        for idx, (dataset, category) in enumerate(categories, 1):
            print(f"\n[{idx}/{total}] Inference: {category}...")
            start = time.time()
            key = f"{dataset}/{category}"
            all_predictions[key] = self.predict(dataset, category, save_json)
            elapsed = time.time() - start
            print(f"✓ [{idx}/{total}] {category} 완료 ({elapsed:.1f}s)")

        logger.info(f"predict_all completed: {total} categories")
        return all_predictions
