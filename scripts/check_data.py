import json
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/samples/fraud_sample.parquet")
REPORT_PATH = Path("reports/data/sample_validation.json")

REQUIRED_COLUMNS = [
    "TransactionID",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "isFraud",
]


def validate_sample(path: Path = DATA_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    df = pd.read_parquet(path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset sample is empty.")

    if not set(df["isFraud"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column isFraud must be binary.")

    fraud_rate = float(df["isFraud"].mean())

    report = {
        "path": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "fraud_rate": fraud_rate,
        "missing_columns": missing_columns,
        "required_columns": REQUIRED_COLUMNS,
        "status": "passed",
    }

    return report


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = validate_sample()
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
