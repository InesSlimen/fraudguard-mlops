import json
import os
from pathlib import Path

from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraudguard-risk-model")

MIN_ROC_AUC = float(os.getenv("PROMOTION_MIN_ROC_AUC", "0.60"))
MIN_PR_AUC = float(os.getenv("PROMOTION_MIN_PR_AUC", "0.03"))
MIN_RECALL = float(os.getenv("PROMOTION_MIN_RECALL", "0.50"))

REPORT_PATH = Path("reports/model/promotion_report.json")


def _get_candidate_version(client: MlflowClient):
    try:
        candidate = client.get_model_version_by_alias(MODEL_NAME, "candidate")
        return candidate.version

    except Exception as err:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        candidate_versions = [
            version for version in versions if version.tags.get("lifecycle") == "candidate"
        ]

        if not candidate_versions:
            raise RuntimeError("No candidate model version found.") from err

        return sorted(candidate_versions, key=lambda version: int(version.version))[-1].version


def _get_run_metrics(client: MlflowClient, model_version: str) -> dict:
    version = client.get_model_version(MODEL_NAME, model_version)
    run_id = version.run_id

    if not run_id:
        run_id = version.tags.get("source_run_id")

    if not run_id:
        raise RuntimeError(f"No run_id found for model version {model_version}")

    run = client.get_run(run_id)
    return run.data.metrics


def evaluate_promotion(metrics: dict) -> tuple[bool, list[str]]:
    failures = []

    if metrics.get("roc_auc", 0.0) < MIN_ROC_AUC:
        failures.append(f"roc_auc below threshold: {metrics.get('roc_auc')} < {MIN_ROC_AUC}")

    if metrics.get("pr_auc", 0.0) < MIN_PR_AUC:
        failures.append(f"pr_auc below threshold: {metrics.get('pr_auc')} < {MIN_PR_AUC}")

    if metrics.get("recall", 0.0) < MIN_RECALL:
        failures.append(f"recall below threshold: {metrics.get('recall')} < {MIN_RECALL}")

    return len(failures) == 0, failures


def promote_candidate(dry_run: bool = True) -> dict:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    candidate_version = _get_candidate_version(client)
    metrics = _get_run_metrics(client, candidate_version)

    approved, failures = evaluate_promotion(metrics)

    report = {
        "model_name": MODEL_NAME,
        "candidate_version": candidate_version,
        "dry_run": dry_run,
        "approved": approved,
        "failures": failures,
        "thresholds": {
            "min_roc_auc": MIN_ROC_AUC,
            "min_pr_auc": MIN_PR_AUC,
            "min_recall": MIN_RECALL,
        },
        "metrics": metrics,
    }

    if approved and not dry_run:
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=candidate_version,
            key="lifecycle",
            value="champion",
        )

        try:
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="champion",
                version=candidate_version,
            )
        except Exception:
            pass

        report["status"] = "promoted"
    elif approved:
        report["status"] = "approved_dry_run"
    else:
        report["status"] = "rejected"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    if not approved:
        raise RuntimeError(json.dumps(report, indent=2))

    return report


def main() -> None:
    dry_run = os.getenv("PROMOTION_DRY_RUN", "true").lower() == "true"
    report = promote_candidate(dry_run=dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()