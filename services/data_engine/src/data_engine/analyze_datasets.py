from __future__ import annotations

from pathlib import Path

from data_engine.statistics import MarketStatisticalAnalyzer


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
    / "statistics"
)


def main() -> None:
    analyzer = MarketStatisticalAnalyzer()

    reports = analyzer.analyze_directory(
        INPUT_DIRECTORY,
        OUTPUT_DIRECTORY,
    )

    print("Statistical analysis completed.")
    print()

    for filename, report in reports.items():
        print(
            f"{filename}: "
            f"rows={report.rows} | "
            f"features={report.feature_count} | "
            f"mean_return={report.mean_return:.6f} | "
            f"return_std={report.return_std:.6f} | "
            f"mean_volatility={report.mean_volatility:.6f} | "
            f"positive_target_ratio="
            f"{report.positive_target_ratio:.4f}"
        )

    print()
    print(
        f"Output directory: {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()