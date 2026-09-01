from __future__ import annotations

from pathlib import Path

from data_engine.pca import MarketPCA, PCAConfig


PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "features"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "pca"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "pca"
)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for input_path in sorted(
        INPUT_DIRECTORY.glob("*.csv")
    ):
        dataframe = __import__(
            "pandas"
        ).read_csv(input_path)

        pca = MarketPCA(
            PCAConfig(
                n_components=0.95
            )
        )

        transformed, report = (
            pca.fit_transform(dataframe)
        )

        output_path = (
            OUTPUT_DIRECTORY
            / input_path.name
        )

        transformed.to_csv(
            output_path,
            index=False,
        )

        variance_path = (
            OUTPUT_DIRECTORY
            / f"{input_path.stem}_variance.csv"
        )

        pca.explained_variance().to_csv(
            variance_path,
            index=False,
        )

        loadings_path = (
            OUTPUT_DIRECTORY
            / f"{input_path.stem}_loadings.csv"
        )

        pca.component_loadings().to_csv(
            loadings_path
        )

        model_path = (
            MODEL_DIRECTORY
            / f"{input_path.stem}_pca.joblib"
        )

        pca.save(model_path)

        final_variance = (
            report.cumulative_explained_variance[-1]
        )

        print(
            f"{input_path.name}: "
            f"{report.input_features} features -> "
            f"{report.output_components} PCs | "
            f"explained={final_variance:.4f}"
        )

    print()
    print(
        f"PCA datasets: {OUTPUT_DIRECTORY}"
    )

    print(
        f"PCA models: {MODEL_DIRECTORY}"
    )


if __name__ == "__main__":
    main()