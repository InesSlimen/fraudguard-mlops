import json
from pathlib import Path

import pandas as pd

from products.fraudguard.features.feature_contract import (
    FEATURES,
    REQUIRED_COLUMNS,
    TARGET,
    validate_feature_columns,
)

INPUT_PATH = Path("data/samples/fraud_sample.parquet")
OUTPUT_DIR = Path("data/processed")
REPORT_PATH = Path("reports/features/feature_contract.json")


def build_feature_dataset(input_path: Path = INPUT_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_path}")

    df = pd.read_parquet(input_path)
    validate_feature_columns(list(df.columns))

    feature_df = df[FEATURES].copy()
    target = df[TARGET].copy()

    return feature_df, target


def write_feature_report(input_path: Path = INPUT_PATH) -> dict:
    df = pd.read_parquet(input_path)
    validate_feature_columns(list(df.columns))

    report = {
        "input_path": str(input_path),
        "rows": int(len(df)),
        "required_columns": REQUIRED_COLUMNS,
        "features": FEATURES,
        "target": TARGET,
        "missing_values": {
            column: int(df[column].isna().sum())
            for column in REQUIRED_COLUMNS
            if column in df.columns
        },
        "status": "passed",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = build_feature_dataset(INPUT_PATH)

    X.to_parquet(OUTPUT_DIR / "features.parquet", index=False)
    y.to_frame(name=TARGET).to_parquet(OUTPUT_DIR / "target.parquet", index=False)

    report = write_feature_report(INPUT_PATH)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()