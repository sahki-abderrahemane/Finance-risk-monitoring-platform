from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from services.data_engine.src.data_engine.validator import MarketDataValidator


RAW_DATA_DIR = Path("ml/datasets/raw")


def main() -> None:
    validator = MarketDataValidator()

    profiles = validator.profile_directory(RAW_DATA_DIR) if hasattr(
        validator,
        "profile_directory",
    ) else None

    if profiles is not None:
        raise RuntimeError(
            "Unexpected profiler API detected. "
            "Use MarketDataValidator.validate() directly."
        )

    from data_engine.loader import CSVMarketDataLoader

    loader = CSVMarketDataLoader()

    files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not files:
        raise RuntimeError(
            f"No CSV files found in {RAW_DATA_DIR}"
        )

    total_valid = 0
    total_invalid = 0

    print("=" * 72)
    print("SENTINEL-AI — RAW DATA VALIDATION REPORT")
    print("=" * 72)

    for path in files:
        print()
        print(f"Asset: {path.stem.upper()}")
        print(f"File:  {path}")

        try:
            dataframe = loader.load(path)
        except Exception as exc:
            total_invalid += 1

            print("Status: INVALID")
            print(f"Loader error: {exc}")
            continue

        report = validator.validate(dataframe)

        print(f"Rows:   {report.rows}")
        print(f"Status: {'VALID' if report.valid else 'INVALID'}")

        if report.valid:
            total_valid += 1
        else:
            total_invalid += 1

            print("Issues:")

            for issue in report.issues:
                print(
                    f"  - [{issue.rule}] "
                    f"{issue.message}"
                )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Datasets checked: {len(files)}")
    print(f"Valid datasets:   {total_valid}")
    print(f"Invalid datasets: {total_invalid}")
    print("=" * 72)


if __name__ == "__main__":
    main()