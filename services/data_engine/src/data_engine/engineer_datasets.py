from __future__ import annotations

from pathlib import Path

from data_engine.features import MarketFeatureEngineer


PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_DIRECTORY = PROJECT_ROOT / "ml" / "datasets" / "processed"
OUTPUT_DIRECTORY = PROJECT_ROOT / "ml" / "datasets" / "features"


def main() -> None:
    engineer = MarketFeatureEngineer()

    reports = engineer.transform_directory(
        INPUT_DIRECTORY,
        OUTPUT_DIRECTORY,
    )

    print("Feature engineering completed.")
    print()

    for filename, report in reports.items():
        print(
            f"{filename}: "
            f"{report.input_rows} -> {report.output_rows} rows | "
            f"removed={report.rows_removed}"
        )

    print()
    print(f"Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()