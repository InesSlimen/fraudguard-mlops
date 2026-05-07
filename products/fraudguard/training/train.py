import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from products.fraudguard.evaluation.metrics import compute_binary_classification_metrics
from products.fraudguard.evaluation.thresholding import find_threshold_for_recall
from products.fraudguard.features.build_features import build_feature_dataset
from products.fraudguard.features.preprocessors import build_preprocessor

DATA_PATH = Path("data/samples/fraud_sample.parquet")
ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports/model")

MODEL_PATH = ARTIFACT_DIR / "fraud_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REPORT_METRICS_PATH = REPORT_DIR / "metrics.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraudguard-baseline")


def train_model(data_path: Path = DATA_PATH) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = build_feature_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_scores = pipeline.predict_proba(X_test)[:, 1]

    threshold = find_threshold_for_recall(y_test, y_scores, min_recall=0.70)

    metrics = compute_binary_classification_metrics(
        y_true=y_test,
        y_scores=y_scores,
        threshold=threshold,
    )

    metrics.update(
        {
            "model_type": "LightGBM",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_rate": float(pd.Series(y).mean()),
        }
    )

    joblib.dump(pipeline, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    REPORT_METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="fraudguard-lightgbm-baseline") as run:
            mlflow.log_params(
                {
                    "model_type": "LightGBM",
                    "n_estimators": 150,
                    "learning_rate": 0.05,
                    "class_weight": "balanced",
                    "threshold": metrics["threshold"],
                }
            )

            mlflow.log_metrics(
                {
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "positive_rate": metrics["positive_rate"],
                }
            )

            mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            metrics["mlflow_run_id"] = run.info.run_id

            METRICS_PATH.write_text(json.dumps(metrics, indent=2))
            REPORT_METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    metrics = train_model()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()