import json
import sys
from pathlib import Path

DEFAULT_METRICS_PATH = Path("reports/model/metrics.json")

MIN_ROC_AUC = 0.60
MIN_PR_AUC = 0.03


def check_metrics(metrics_path: Path = DEFAULT_METRICS_PATH) -> dict:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())

    required_metrics = ["roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]

    missing = [metric for metric in required_metrics if metric not in metrics]

    if missing:
        raise ValueError(f"Missing required metrics: {missing}")

    failures = []

    if metrics["roc_auc"] < MIN_ROC_AUC:
        failures.append(f"roc_auc {metrics['roc_auc']} < {MIN_ROC_AUC}")

    if metrics["pr_auc"] < MIN_PR_AUC:
        failures.append(f"pr_auc {metrics['pr_auc']} < {MIN_PR_AUC}")

    report = {
        "status": "passed" if not failures else "failed",
        "thresholds": {
            "min_roc_auc": MIN_ROC_AUC,
            "min_pr_auc": MIN_PR_AUC,
        },
        "metrics": metrics,
        "failures": failures,
    }

    if failures:
        raise ValueError(json.dumps(report, indent=2))

    return report


def main() -> None:
    metrics_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_METRICS_PATH
    report = check_metrics(metrics_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
