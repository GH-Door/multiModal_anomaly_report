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

import torch
from anomalib.models import Patchcore, WinClip, EfficientAd
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
from anomalib.engine import Engine
from pytorch_lightning.callbacks import Callback

# PyTorch 2.6+ weights_only=True 대응: Anomalib 클래스 허용
torch.serialization.add_safe_globals([EfficientAdModelSize])

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

    def get_engine(self, dataset: str = None, category: str = None, model=None, datamodule=None, stage: str = None):
        # Import necessary callbacks
        from pytorch_lightning.callbacks import ModelCheckpoint

        # WandB logger 설정 (predict 시에는 비활성화)
        logger_config = self.engine_config.get("logger", False)
        if logger_config == "wandb":
            if stage == "predict" or not (dataset and category):
                # predict 또는 dataset/category 없으면 wandb 비활성화
                logger_config = False
            else:
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

        # --- Custom Callbacks for Checkpoint and Visualizer ---

        # 1. ModelCheckpoint Callback (for simplified path)
        if dataset and category: # Only add if dataset and category are available for path
            checkpoint_dir = (
                self.output_root
                / self.MODEL_DIR_MAP.get(self.model_name, self.model_name.capitalize())
                / dataset
                / category
                / "v0" # Version folder
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model_checkpoint_callback = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="model", # Saves as model.ckpt
                save_last=True,
                save_top_k=1,
                monitor="image_AUROC", # Assuming this is a common metric to monitor
                mode="max",
            )
            callbacks.append(model_checkpoint_callback)
        # If dataset/category not available, default ModelCheckpoint might still be added by Anomalib.
        # Or, if this is a predict-only scenario without training, no checkpoint is needed.

        # Visualizer Callback (yaml에서 visualizer: true일 때만)
        visualizer_enabled = self.model_params.get("visualizer", False)
        if visualizer_enabled and stage == "predict" and dataset and category:
            try:
                # Anomalib 버전에 따라 import 경로가 다름
                try:
                    from anomalib.callbacks import ImageVisualizerCallback as VisualizerCallback
                except ImportError:
                    try:
                        from anomalib.utils.callbacks import ImageVisualizerCallback as VisualizerCallback
                    except ImportError:
                        VisualizerCallback = None

                if VisualizerCallback:
                    image_save_path = (
                        self.output_root
                        / self.MODEL_DIR_MAP.get(self.model_name, self.model_name.capitalize())
                        / dataset
                        / category
                        / "predictions"
                    )
                    image_save_path.mkdir(parents=True, exist_ok=True)
                    callbacks.append(VisualizerCallback(image_save_path=str(image_save_path)))
            except Exception as e:
                logger.warning(f"Visualizer callback not available: {e}")

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

    # Anomalib이 저장하는 실제 폴더명 매핑
    MODEL_DIR_MAP = {
        "patchcore": "Patchcore",
        "winclip": "WinClip",
        "efficientad": "EfficientAd",
    }

    def get_ckpt_path(self, dataset: str, category: str) -> Path | None:
        if self.model_name == "winclip":
            return None

        model_dir = self.MODEL_DIR_MAP.get(self.model_name, self.model_name.capitalize())

        # Simplified checkpoint path as per user's request:
        # output/model_name/dataset/category/v0/model.ckpt
        return (
            self.output_root
            / model_dir
            / dataset
            / category
            / "v0"
            / "model.ckpt"
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
        ckpt_path = self.get_ckpt_path(dataset, category)
        engine.fit(datamodule=datamodule, model=model, ckpt_path=str(ckpt_path) if ckpt_path else None)

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
        dm_kwargs["include_mask"] = True  # predict 시 GT mask 포함
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        engine = self.get_engine(dataset, category, model=model, datamodule=datamodule, stage="predict")
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
        """Get list of (dataset, category) tuples from DATASETS."""
        return [
            (dataset, category)
            for dataset in self.loader.DATASETS
            for category in self.loader.get_categories(dataset)
        ]

    def get_trained_categories(self) -> list[tuple[str, str]]:
        """Get list of (dataset, category) tuples that have trained checkpoints."""
        model_dir = self.MODEL_DIR_MAP.get(self.model_name, self.model_name.capitalize())
        model_path = self.output_root / model_dir

        if not model_path.exists():
            return []

        trained = []
        for dataset_dir in sorted(model_path.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            for category_dir in sorted(dataset_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                category = category_dir.name
                ckpt = category_dir / "v0/model.ckpt"
                if ckpt.exists():
                    trained.append((dataset, category))
        return trained

    def fit_all(self):
        categories = self.get_all_categories()
        total = len(categories)
        logger.info(f"Starting fit_all: {total} categories")

        for idx, (dataset, category) in enumerate(categories, 1):
            msg_start = f"[{idx}/{total}] Training: {dataset}/{category}..."
            print(f"\n{msg_start}")
            logger.info(msg_start)
            start = time.time()
            self.fit(dataset, category)
            elapsed = time.time() - start
            msg_done = f"[{idx}/{total}] {dataset}/{category} done ({elapsed:.1f}s)"
            print(f"✓ {msg_done}")
            logger.info(msg_done)

        logger.info(f"fit_all completed: {total} categories")

    def predict_all(self, save_json: bool = None):
        categories = self.get_trained_categories()
        total = len(categories)
        logger.info(f"Starting predict_all: {total} trained categories")

        all_predictions = {}
        for idx, (dataset, category) in enumerate(categories, 1):
            msg_start = f"[{idx}/{total}] Inference: {dataset}/{category}..."
            print(f"\n{msg_start}")
            logger.info(msg_start)
            start = time.time()
            key = f"{dataset}/{category}"
            all_predictions[key] = self.predict(dataset, category, save_json)
            elapsed = time.time() - start
            msg_done = f"[{idx}/{total}] {dataset}/{category} done ({elapsed:.1f}s)"
            print(f"✓ {msg_done}")
            logger.info(msg_done)

        logger.info(f"predict_all completed: {total} categories")
        return all_predictions
