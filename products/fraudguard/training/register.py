import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from products.fraudguard.training.select_best_run import select_best_run

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraudguard-risk-model")
REPORT_PATH = Path("reports/model/registration_report.json")


def register_candidate(run_id: str | None = None) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    selected_run_id = run_id or select_best_run()
    model_uri = f"runs:/{selected_run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=result.version,
        key="lifecycle",
        value="candidate",
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=result.version,
        key="source_run_id",
        value=selected_run_id,
    )

    try:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="candidate",
            version=result.version,
        )
    except Exception:
        # Older MLflow setups may not support aliases.
        pass

    report = {
        "status": "registered",
        "model_name": MODEL_NAME,
        "model_version": result.version,
        "run_id": selected_run_id,
        "model_uri": model_uri,
        "alias": "candidate",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    report = register_candidate()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()