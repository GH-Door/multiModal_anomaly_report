#!/usr/bin/env python
"""Quick debug script to verify training pipeline works correctly.

Tests the ACTUAL training modules (train_anomalib.py, dataloader.py).

Usage:
    # Quick check: verify model + datamodule creation
    python scripts/debug_pipeline.py --mode check

    # Training test: run minimal training with actual Anomalibs class
    python scripts/debug_pipeline.py --mode train

    # Full test: check + train + inference
    python scripts/debug_pipeline.py --mode all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(PROJ_ROOT))


def check_config_and_model():
    """Check that config is loaded correctly and model is created with correct PreProcessor."""
    print("=" * 60)
    print("1. Checking config and model creation (actual modules)")
    print("=" * 60)

    # Import actual training module
    from scripts.train_anomalib import Anomalibs

    runner = Anomalibs(config_path="configs/anomaly.yaml")

    print(f"  Model: {runner.model_name}")
    print(f"  Image size (from config): {runner.image_size}")
    print(f"  Device: {runner.device}")
    print(f"  Accelerator: {runner.accelerator}")

    # Create model and check PreProcessor
    model = runner.get_model()
    print(f"  Model PreProcessor: {model.pre_processor.transform}")

    # Verify it's Normalize only (not Resize+Normalize)
    transform_str = str(model.pre_processor.transform)
    if "Resize" in transform_str:
        print("  [WARNING] Model PreProcessor contains Resize - should be Normalize only!")
    else:
        print("  [OK] Model PreProcessor is Normalize only (Resize handled by DataModule)")

    return runner


def check_datamodule(runner):
    """Check DataModule creation and verify image sizes."""
    print("\n" + "=" * 60)
    print("2. Checking DataModule (actual modules)")
    print("=" * 60)

    # Find first valid dataset/category
    datasets = runner.config.get("data", {}).get("datasets", [])
    config_categories = runner.config.get("data", {}).get("categories", [])

    dataset = None
    category = None
    for ds in datasets:
        available_cats = runner.loader.get_categories(ds)
        if config_categories:
            valid_cats = [c for c in config_categories if c in available_cats]
        else:
            valid_cats = available_cats
        if valid_cats:
            dataset = ds
            category = valid_cats[0]
            break

    if not dataset or not category:
        print("  [ERROR] No valid dataset/category found")
        return None, None

    print(f"  Using: {dataset}/{category}")

    # Create datamodule using actual loader
    dm_kwargs = runner.get_datamodule_kwargs()
    dm = runner.loader.get_datamodule(dataset, category, **dm_kwargs)

    print(f"  DataModule image_size: {dm.image_size}")
    print(f"  DataModule transform: {dm.transform}")

    # Setup and verify
    dm.setup(stage="fit")

    if dm.train_data is None or len(dm.train_data) == 0:
        print(f"  [ERROR] No training data found!")
        return None, None

    print(f"  Training samples: {len(dm.train_data)}")

    # Check actual tensor size
    sample = dm.train_data[0]
    tensor_shape = sample.image.shape
    print(f"  Sample tensor shape: {tensor_shape}")

    expected_size = runner.image_size
    actual_size = tuple(tensor_shape[-2:])

    if actual_size != expected_size:
        print(f"  [ERROR] Expected {expected_size}, got {actual_size}")
        return None, None

    print(f"  [OK] DataModule correctly produces {actual_size} tensors")
    return dataset, category


def run_training_test(runner, dataset: str, category: str):
    """Run actual training using Anomalibs.fit()."""
    print("\n" + "=" * 60)
    print("3. Running training test (actual Anomalibs.fit)")
    print("=" * 60)

    print(f"  Training: {dataset}/{category}")
    print(f"  Image size: {runner.image_size}")

    # Run actual training
    try:
        runner.fit(dataset, category)
        print("  [OK] Training completed successfully")
        return True
    except Exception as e:
        print(f"  [ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_checkpoint(runner, dataset: str, category: str):
    """Verify checkpoint was saved correctly."""
    print("\n" + "=" * 60)
    print("4. Checking saved checkpoint")
    print("=" * 60)

    ckpt_path = runner.get_ckpt_path(dataset, category)

    if ckpt_path is None or not ckpt_path.exists():
        print(f"  [ERROR] Checkpoint not found")
        return False

    print(f"  Checkpoint: {ckpt_path}")

    # Load and verify
    from anomalib.models import Patchcore
    model = Patchcore.load_from_checkpoint(str(ckpt_path), map_location="cpu", weights_only=False)

    print(f"  Loaded PreProcessor: {model.pre_processor.transform}")

    # Check memory bank
    if hasattr(model, 'model') and hasattr(model.model, 'memory_bank'):
        mb_shape = model.model.memory_bank.shape
        print(f"  Memory bank shape: {mb_shape}")

        if mb_shape[0] == 0:
            print("  [ERROR] Memory bank is empty!")
            return False

    print("  [OK] Checkpoint is valid")
    return True


def run_inference_test(runner, dataset: str, category: str):
    """Test inference produces valid anomaly scores."""
    print("\n" + "=" * 60)
    print("5. Running inference test")
    print("=" * 60)

    # Get a test image
    data_root = runner.data_root
    test_dir = data_root / dataset / category / "test" / "good"

    if not test_dir.exists():
        print(f"  [WARNING] Test directory not found: {test_dir}")
        return True  # Not a failure

    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not test_images:
        print("  [WARNING] No test images found")
        return True

    test_image = test_images[0]
    print(f"  Test image: {test_image}")

    # Load model
    import cv2
    import numpy as np
    from anomalib.models import Patchcore

    ckpt_path = runner.get_ckpt_path(dataset, category)
    model = Patchcore.load_from_checkpoint(str(ckpt_path), map_location="cpu", weights_only=False)
    model.eval()

    # Preprocess (same as run_ad_inference_ckpt.py)
    img = cv2.imread(str(test_image))
    h, w = runner.image_size
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    print(f"  Input tensor shape: {tensor.shape}")

    # Inference
    with torch.no_grad():
        outputs = model(tensor)

    pred_score = outputs.pred_score.item()
    print(f"  Anomaly score: {pred_score:.4f}")

    # Check anomaly map
    if outputs.anomaly_map is not None:
        amap = outputs.anomaly_map[0, 0].numpy()
        print(f"  Anomaly map: shape={amap.shape}, min={amap.min():.4f}, max={amap.max():.4f}, mean={amap.mean():.4f}")

        if amap.min() == amap.max():
            print("  [ERROR] Anomaly map has constant values!")
            return False

    # Score should be reasonable (not 0 or 1 exactly for normal image)
    if pred_score == 0.0 or pred_score == 1.0:
        print(f"  [WARNING] Score is exactly {pred_score} - might indicate preprocessing issue")

    print("  [OK] Inference produces valid results")
    return True


def main():
    parser = argparse.ArgumentParser(description="Debug training pipeline")
    parser.add_argument("--mode", type=str, default="check",
                        choices=["check", "train", "all"],
                        help="Debug mode: check (verify config/model), train (run training), all (full test)")
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml",
                        help="Config file path")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PIPELINE DEBUG TEST")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")

    # Step 1: Check config and model
    runner = check_config_and_model()

    if args.mode == "check":
        # Also check datamodule
        check_datamodule(runner)
        print("\n[DONE] Basic checks completed")
        return

    # Step 2: Check DataModule
    dataset, category = check_datamodule(runner)
    if not dataset:
        print("\n[FAILED] DataModule check failed")
        return

    if args.mode in ["train", "all"]:
        # Step 3: Run training
        success = run_training_test(runner, dataset, category)
        if not success:
            print("\n[FAILED] Training failed")
            return

        # Step 4: Check checkpoint
        success = check_checkpoint(runner, dataset, category)
        if not success:
            print("\n[FAILED] Checkpoint check failed")
            return

        if args.mode == "all":
            # Step 5: Inference test
            success = run_inference_test(runner, dataset, category)
            if not success:
                print("\n[FAILED] Inference test failed")
                return

    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
