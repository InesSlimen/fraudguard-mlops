import json
import os
from pathlib import Path

from mlflow import MlflowClient

REPORT_PATH = Path("reports/mlflow/mlflow_smoke.txt")


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraudguard-baseline")

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=5,
        order_by=["start_time DESC"],
    )

    if not runs:
        raise RuntimeError("No MLflow runs found.")

    latest = runs[0]

    report = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment.experiment_id,
        "latest_run_id": latest.info.run_id,
        "status": "passed",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()