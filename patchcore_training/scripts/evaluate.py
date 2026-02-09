#!/usr/bin/env python
"""Evaluate PatchCore models.

Usage:
    # Evaluate all trained models
    python patchcore_training/scripts/evaluate.py

    # Evaluate specific category
    python patchcore_training/scripts/evaluate.py --dataset GoodsAD --category cigarette_box

    # Save results to file
    python patchcore_training/scripts/evaluate.py --output output/patchcore_eval_results.json
"""

import argparse
import sys
from pathlib import Path

# Add patchcore_training to path
SCRIPT_DIR = Path(__file__).resolve().parent
PATCHCORE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PATCHCORE_ROOT))

from src.trainer import PatchCoreTrainer
from src.evaluator import PatchCoreEvaluator
from src.utils import load_config, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore models")

    parser.add_argument(
        "--config",
        type=str,
        default=str(PATCHCORE_ROOT / "config" / "config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Evaluate only specific dataset",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Evaluate only specific category (requires --dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    setup_seed(args.seed)

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Create trainer and evaluator
    trainer = PatchCoreTrainer(config)
    evaluator = PatchCoreEvaluator(config)

    # Evaluate
    if args.dataset and args.category:
        # Evaluate single category
        category_key = f"{args.dataset}/{args.category}"
        print(f"\nEvaluating: {category_key}")
        model = trainer.load_model(args.dataset, args.category)
        if model is None:
            print(f"Model not found for {category_key}")
            return

        metrics = evaluator.evaluate_category(
            model=model,
            dataset_name=args.dataset,
            category=args.category,
        )
        results = {category_key: metrics}
    else:
        # Load all models
        models = trainer.load_all_models()
        if not models:
            print("No trained models found!")
            return

        # Filter by dataset if specified
        if args.dataset:
            models = {k: v for k, v in models.items() if k.startswith(args.dataset + "/")}

        # Evaluate all
        results = evaluator.evaluate_all(models)

    # Save results if output path specified
    if args.output:
        evaluator.save_results(results, args.output)
    else:
        # Print per-category results
        print("\n" + "=" * 60)
        print(f"{'Category':<35} {'I-AUROC':>10} {'P-AUROC':>10} {'PRO':>10}")
        print("=" * 60)
        for key, metrics in results.items():
            if key == "average":
                continue
            print(f"{key:<35} {metrics.get('image_auroc', 0):>10.4f} "
                  f"{metrics.get('pixel_auroc', 0):>10.4f} {metrics.get('pro', 0):>10.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
