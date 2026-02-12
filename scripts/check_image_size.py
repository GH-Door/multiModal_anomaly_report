#!/usr/bin/env python
"""Quick check: config image_size -> model input size.

Usage:
    python scripts/check_image_size.py
    python scripts/check_image_size.py --config configs/anomaly.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torchvision.transforms import v2
from anomalib.models import Patchcore
from anomalib.pre_processing import PreProcessor
from src.utils.loaders import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anomaly.yaml")
    args = parser.parse_args()

    print("=" * 50)
    print("IMAGE SIZE CHECK")
    print("=" * 50)

    # 1. Config 로드
    config = load_config(args.config)
    config_size = tuple(config.get("data", {}).get("image_size", [256, 256]))
    print(f"\n[1] Config file: {args.config}")
    print(f"    data.image_size = {config_size}")

    # 2. PreProcessor 생성 (train_anomalib.py와 동일 로직)
    transform = v2.Compose([
        v2.Resize(config_size, antialias=True),
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pre_processor = PreProcessor(transform=transform)

    # 3. 모델 생성
    model = Patchcore(pre_processor=pre_processor)
    model.eval()

    # 4. 모델 내부 PreProcessor 확인
    model_resize = model.pre_processor.transform.transforms[0].size
    print(f"\n[2] Model PreProcessor")
    print(f"    Resize size = {model_resize}")

    # 5. 실제 텐서 테스트
    print(f"\n[3] Tensor Flow Test")
    dummy = torch.rand(1, 3, 1280, 1600)  # 원본 이미지 크기 가정
    print(f"    Input shape:  {list(dummy.shape)}")

    output = model.pre_processor(dummy)
    print(f"    Output shape: {list(output.shape)}")

    # 6. 결과
    print("\n" + "=" * 50)
    if model_resize == list(config_size) and output.shape[-2:] == torch.Size(config_size):
        print(f"✅ SUCCESS: Model uses {config_size} as configured!")
    else:
        print(f"❌ FAIL: Config={config_size}, Model={model_resize}")
    print("=" * 50)


if __name__ == "__main__":
    main()
