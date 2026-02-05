from pathlib import Path
from typing import Generator

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from anomalib.data import MVTecAD, Visa, Folder
from anomalib.data.datamodules.base import AnomalibDataModule
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.dataclasses import ImageBatch, ImageItem
from anomalib.data.utils.image import read_image, read_mask


class MVTecLOCODataset(AnomalibDataset):
    """MVTec-LOCO 커스텀 Dataset (중첩 마스크 구조 처리)"""
    def __init__(
        self,
        root: Path | str,
        category: str,
        split: str = "train",
    ):
        super().__init__(augmentations=None)
        self.root = Path(root)
        self._category = category
        self.split = split
        self._samples = self.make_dataset()

    def make_dataset(self) -> pd.DataFrame:
        """MVTec-LOCO 샘플 DataFrame 생성"""
        cat_path = self.root / self.category
        samples_list = []

        if self.split == "train":
            train_good = cat_path / "train" / "good"
            if train_good.exists():
                for img_path in sorted(train_good.glob("*.png")):
                    if img_path.name.startswith("."):
                        continue
                    samples_list.append({
                        "image_path": str(img_path),
                        "label": "normal",
                        "label_index": 0,
                        "mask_path": None,
                        "split": "train",
                    })
        else:
            test_dir = cat_path / "test"
            gt_dir = cat_path / "ground_truth"

            if test_dir.exists():
                for defect_dir in sorted(test_dir.iterdir()):
                    if not defect_dir.is_dir() or defect_dir.name.startswith("."):
                        continue

                    defect_type = defect_dir.name
                    is_normal = defect_type == "good"

                    for img_path in sorted(defect_dir.glob("*.png")):
                        if img_path.name.startswith("."):
                            continue

                        mask_path = None
                        if not is_normal:
                            stem = img_path.stem
                            mask_dir = gt_dir / defect_type / stem
                            # MVTec-LOCO: 마스크 폴더 내 파일명은 000.png, 001.png, ... (이미지 stem과 다름)
                            if mask_dir.exists() and mask_dir.is_dir():
                                mask_files = sorted(mask_dir.glob("*.png"))
                                if mask_files:
                                    # 첫 번째 마스크 사용 (여러 개면 나중에 합칠 수 있음)
                                    mask_path = str(mask_files[0])

                        samples_list.append({
                            "image_path": str(img_path),
                            "label": "normal" if is_normal else "anomaly",
                            "label_index": 0 if is_normal else 1,
                            "mask_path": mask_path,
                            "split": "test",
                        })

        samples = pd.DataFrame(samples_list)
        samples.attrs["task"] = "segmentation"
        return samples

    def __getitem__(self, index: int) -> ImageItem:
        """마스크가 없는 샘플을 안전하게 처리"""
        sample = self.samples.iloc[index]
        image_path = sample.image_path
        mask_path = sample.mask_path
        label_index = sample.label_index

        image = read_image(image_path, as_tensor=True)

        # mask_path가 유효한 문자열인 경우에만 마스크 로드
        gt_mask = None
        if pd.notna(mask_path) and isinstance(mask_path, str) and mask_path != "":
            gt_mask = read_mask(mask_path, as_tensor=True)

        # augmentations 적용
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=gt_mask) if gt_mask is not None else self.augmentations(image=image)
            image = augmented["image"]
            if gt_mask is not None and "mask" in augmented:
                gt_mask = augmented["mask"]

        # mask가 None이면 이미지 크기에 맞는 빈 마스크 생성 (collate 호환)
        if gt_mask is None:
            gt_mask = torch.zeros(image.shape[1:], dtype=torch.float32)

        item = ImageItem(
            image_path=image_path,
            image=image,
            gt_label=torch.tensor(label_index),
            gt_mask=gt_mask,
        )
        return item


class GoodsADDataset(AnomalibDataset):
    """GoodsAD 커스텀 Dataset (이미지 jpg, 마스크 png 처리)"""
    def __init__(
        self,
        root: Path | str,
        category: str,
        split: str = "train",
    ):
        super().__init__(augmentations=None)
        self.root = Path(root)
        self._category = category
        self.split = split
        self._samples = self.make_dataset()

    def make_dataset(self) -> pd.DataFrame:
        """GoodsAD 샘플 DataFrame 생성"""
        cat_path = self.root / self.category
        samples_list = []

        if self.split == "train":
            train_good = cat_path / "train" / "good"
            if train_good.exists():
                for img_path in sorted(train_good.glob("*.jpg")):
                    if img_path.name.startswith("."):
                        continue
                    samples_list.append({
                        "image_path": str(img_path),
                        "label": "normal",
                        "label_index": 0,
                        "mask_path": None,
                        "split": "train",
                    })
        else:
            test_dir = cat_path / "test"
            gt_dir = cat_path / "ground_truth"

            if test_dir.exists():
                for defect_dir in sorted(test_dir.iterdir()):
                    if not defect_dir.is_dir() or defect_dir.name.startswith("."):
                        continue

                    defect_type = defect_dir.name
                    is_normal = defect_type == "good"

                    for img_path in sorted(defect_dir.glob("*.jpg")):
                        if img_path.name.startswith("."):
                            continue

                        mask_path = None
                        if not is_normal:
                            # GoodsAD: 마스크는 .png, 이미지와 동일한 stem
                            mask_file = gt_dir / defect_type / f"{img_path.stem}.png"
                            if mask_file.exists():
                                mask_path = str(mask_file)

                        samples_list.append({
                            "image_path": str(img_path),
                            "label": "normal" if is_normal else "anomaly",
                            "label_index": 0 if is_normal else 1,
                            "mask_path": mask_path,
                            "split": "test",
                        })

        samples = pd.DataFrame(samples_list)
        samples.attrs["task"] = "segmentation"
        return samples

    def __getitem__(self, index: int) -> ImageItem:
        """마스크가 없는 샘플을 안전하게 처리"""
        sample = self.samples.iloc[index]
        image_path = sample.image_path
        mask_path = sample.mask_path
        label_index = sample.label_index

        image = read_image(image_path, as_tensor=True)

        gt_mask = None
        if pd.notna(mask_path) and isinstance(mask_path, str) and mask_path != "":
            gt_mask = read_mask(mask_path, as_tensor=True)

        if self.augmentations:
            augmented = self.augmentations(image=image, mask=gt_mask) if gt_mask is not None else self.augmentations(image=image)
            image = augmented["image"]
            if gt_mask is not None and "mask" in augmented:
                gt_mask = augmented["mask"]

        # mask가 None이면 이미지 크기에 맞는 빈 마스크 생성 (collate 호환)
        if gt_mask is None:
            gt_mask = torch.zeros(image.shape[1:], dtype=torch.float32)

        item = ImageItem(
            image_path=image_path,
            image=image,
            gt_label=torch.tensor(label_index),
            gt_mask=gt_mask,
        )
        return item


class GoodsADDataModule(LightningDataModule):
    """GoodsAD 커스텀 DataModule"""
    def __init__(
        self,
        root: str | Path,
        category: str,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self._name = "GoodsAD"

        self.train_data = None
        self.test_data = None

    @property
    def name(self) -> str:
        return self._name

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_data = GoodsADDataset(
                root=self.root,
                category=self.category,
                split="train",
            )
            self.test_data = GoodsADDataset(
                root=self.root,
                category=self.category,
                split="test",
            )

        if stage == "test" or stage == "predict":
            if self.test_data is None:
                self.test_data = GoodsADDataset(
                    root=self.root,
                    category=self.category,
                    split="test",
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate,
        )


class MVTecLOCODataModule(LightningDataModule):
    """MVTec-LOCO 커스텀 DataModule"""
    def __init__(
        self,
        root: str | Path,
        category: str,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self._name = "MVTec-LOCO"

        # anomalib Engine 호환성
        self.train_data = None
        self.test_data = None

    @property
    def name(self) -> str:
        return self._name

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_data = MVTecLOCODataset(
                root=self.root,
                category=self.category,
                split="train",
            )
            self.test_data = MVTecLOCODataset(
                root=self.root,
                category=self.category,
                split="test",
            )

        if stage == "test" or stage == "predict":
            if self.test_data is None:
                self.test_data = MVTecLOCODataset(
                    root=self.root,
                    category=self.category,
                    split="test",
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate,
        )


class MMADLoader:
    # DATASETS = ["MVTec-LOCO"]  # 단일 Test
    DATASETS = ["GoodsAD", "MVTec-LOCO"]
    # DATASETS = ["MVTec-AD", "VisA", "GoodsAD", "MVTec-LOCO"]

    EXCLUDE_DIRS = {"split_csv", "visa_pytorch"}

    def __init__(self, root: str = "dataset/MMAD"):
        self.root = Path(root)

    def get_categories(self, dataset: str) -> list[str]:
        ds_path = self.root / dataset
        if not ds_path.exists():
            return []
        return sorted([
            d.name for d in ds_path.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.name not in self.EXCLUDE_DIRS
        ])

    def mvtec_ad(self, category: str, **kwargs) -> AnomalibDataModule:
        kwargs.pop("include_mask", None)  # MVTec-AD는 자동으로 mask 로드
        return MVTecAD(
            root=str(self.root / "MVTec-AD"),
            category=category,
            **kwargs
        )

    def visa(self, category: str, **kwargs) -> AnomalibDataModule:
        kwargs.pop("include_mask", None)  # VisA는 자동으로 mask 로드
        return Visa(
            root=str(self.root / "VisA"),
            category=category,
            **kwargs
        )

    def mvtec_loco(self, category: str, **kwargs) -> MVTecLOCODataModule:
        kwargs.pop("include_mask", None)  # MVTec-LOCO는 커스텀 DataModule에서 처리
        return MVTecLOCODataModule(
            root=str(self.root / "MVTec-LOCO"),
            category=category,
            **kwargs
        )

    def goods_ad(self, category: str, **kwargs) -> GoodsADDataModule:
        kwargs.pop("include_mask", None)  # GoodsAD는 커스텀 DataModule에서 mask 처리
        return GoodsADDataModule(
            root=str(self.root / "GoodsAD"),
            category=category,
            **kwargs
        )

    def get_datamodule(self, dataset: str, category: str, **kwargs):
        loaders = {
            "MVTec-AD": self.mvtec_ad,
            "VisA": self.visa,
            "MVTec-LOCO": self.mvtec_loco,
            "GoodsAD": self.goods_ad,
        }

        if dataset not in loaders:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(loaders.keys())}")
        return loaders[dataset](category, **kwargs)

    def iter_all(self, **kwargs) -> Generator[tuple[str, str, LightningDataModule], None, None]:
        for dataset in self.DATASETS:
            categories = self.get_categories(dataset)
            for category in categories:
                datamodule = self.get_datamodule(dataset, category, **kwargs)
                yield dataset, category, datamodule
