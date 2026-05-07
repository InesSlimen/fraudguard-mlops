import json
from pathlib import Path

import pandas as pd
import pandera as pa

from contracts.data_contract import get_data_schema

DATA_PATH = Path("data/samples/fraud_sample.parquet")
REPORT_PATH = Path("reports/data/validation_report.json")


def validate_data(path: Path = DATA_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_parquet(path)
    schema = get_data_schema()

    try:
        schema.validate(df, lazy=True)
        status = "passed"
        errors = None
    except pa.errors.SchemaErrors as e:
        status = "failed"
        errors = e.failure_cases.to_dict(orient="records")

    fraud_rate = float(df["isFraud"].mean()) if "isFraud" in df else None

    report = {
        "path": str(path),
        "rows": len(df),
        "fraud_rate": fraud_rate,
        "status": status,
        "errors": errors,
    }

    return report


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = validate_data()
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))

    if report["status"] != "passed":
        raise ValueError("Data validation failed")


if __name__ == "__main__":
    main()
