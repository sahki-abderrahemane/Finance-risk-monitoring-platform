from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_engine.feature_fusion import (
    DEFAULT_CNN_CHECKPOINT,
    DEFAULT_FINBERT_DIR,
    DEFAULT_FUSION_DATASET,
    DEFAULT_NEWS_DATASET,
    DEFAULT_OUTPUT_DIR,
    FeatureFusionConfig,
    build_fused_splits,
    save_fused_splits,
)
from multimodal_engine.fusion_dataset import (
    load_fusion_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the leakage-safe Sentinel-AI "
            "market + FinBERT + CNN feature representation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_FUSION_DATASET,
    )

    parser.add_argument(
        "--news",
        type=Path,
        default=DEFAULT_NEWS_DATASET,
    )

    parser.add_argument(
        "--finbert",
        type=Path,
        default=DEFAULT_FINBERT_DIR,
    )

    parser.add_argument(
        "--cnn",
        type=Path,
        default=DEFAULT_CNN_CHECKPOINT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--vision-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_fusion_dataset(
        args.dataset
    )

    config = FeatureFusionConfig(
        device=args.device,
        text_batch_size=args.text_batch_size,
        vision_batch_size=args.vision_batch_size,
        num_workers=args.workers,
    )

    (
        train,
        validation,
        test,
        manifest,
        pipeline,
    ) = build_fused_splits(
        dataset,
        news_path=args.news,
        finbert_dir=args.finbert,
        cnn_checkpoint=args.cnn,
        config=config,
    )

    artifacts = save_fused_splits(
        train,
        validation,
        test,
        manifest,
        args.output,
        market_representation=pipeline.market,
    )

    print(
        "FEATURE FUSION COMPLETE"
    )

    print(
        f"Train:      {train.features.shape}"
    )

    print(
        f"Validation: {validation.features.shape}"
    )

    print(
        f"Test:       {test.features.shape}"
    )

    print(
        f"Feature dimensions: {train.dimensions}"
    )

    print()
    print("Representation:")

    print(
        "  market    -> StandardScaler + PCA fitted on train only"
    )

    print(
        "  text      -> mean-pooled FinBERT encoder"
    )

    print(
        "  vision    -> CNN penultimate representation"
    )

    print(
        "  fusion    -> [market || text || vision]"
    )

    print()
    print("Leakage policy:")

    print(
        "  targets/future-derived vision metadata are excluded from inputs"
    )

    print()

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print()
    print("Artifacts:")

    for field, path in artifacts.__dict__.items():
        print(
            f"  {field}: {path}"
        )


if __name__ == "__main__":
    main()