import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
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


def train_and_evaluate_model(
    X_train, X_test, y_train, y_test, model, model_type: str, model_params: dict
) -> dict:
    """Train and evaluate a single model."""
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
            "model_type": model_type,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_rate": float(pd.Series(y_test).mean()),
            "threshold": float(threshold),
        }
    )

    return pipeline, metrics, model_params


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

    # Train LightGBM model
    lgbm_model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    lgbm_params = {
        "model_type": "LightGBM",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "class_weight": "balanced",
    }

    lgbm_pipeline, lgbm_metrics, lgbm_params = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, lgbm_model, "LightGBM", lgbm_params
    )

    # Train RandomForest model
    rf_model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_params = {
        "model_type": "RandomForest",
        "n_estimators": 150,
        "class_weight": "balanced",
    }

    rf_pipeline, rf_metrics, rf_params = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, rf_model, "RandomForest", rf_params
    )

    # Save the best model (based on ROC-AUC)
    best_model = (
        lgbm_pipeline
        if lgbm_metrics["roc_auc"] >= rf_metrics["roc_auc"]
        else rf_pipeline
    )
    best_metrics = (
        lgbm_metrics
        if lgbm_metrics["roc_auc"] >= rf_metrics["roc_auc"]
        else rf_metrics
    )

    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(best_metrics, indent=2))
    REPORT_METRICS_PATH.write_text(json.dumps(best_metrics, indent=2))

    # Prepare comparison results
    comparison_results = {
        "lgbm": lgbm_metrics,
        "random_forest": rf_metrics,
        "best_model": best_metrics["model_type"],
    }

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Log LightGBM
        with mlflow.start_run(run_name="fraudguard-lightgbm-baseline") as run:
            mlflow.log_params(
                {
                    **lgbm_params,
                    "threshold": lgbm_metrics["threshold"],
                }
            )
            mlflow.log_metrics(
                {
                    "roc_auc": lgbm_metrics["roc_auc"],
                    "pr_auc": lgbm_metrics["pr_auc"],
                    "precision": lgbm_metrics["precision"],
                    "recall": lgbm_metrics["recall"],
                    "f1": lgbm_metrics["f1"],
                    "positive_rate": lgbm_metrics["positive_rate"],
                }
            )
            mlflow.sklearn.log_model(lgbm_pipeline, artifact_path="model")
            lgbm_metrics["mlflow_run_id"] = run.info.run_id

        # Log RandomForest
        with mlflow.start_run(run_name="fraudguard-randomforest-baseline") as run:
            mlflow.log_params(
                {
                    **rf_params,
                    "threshold": rf_metrics["threshold"],
                }
            )
            mlflow.log_metrics(
                {
                    "roc_auc": rf_metrics["roc_auc"],
                    "pr_auc": rf_metrics["pr_auc"],
                    "precision": rf_metrics["precision"],
                    "recall": rf_metrics["recall"],
                    "f1": rf_metrics["f1"],
                    "positive_rate": rf_metrics["positive_rate"],
                }
            )
            mlflow.sklearn.log_model(rf_pipeline, artifact_path="model")
            rf_metrics["mlflow_run_id"] = run.info.run_id

        METRICS_PATH.write_text(json.dumps(best_metrics, indent=2))
        REPORT_METRICS_PATH.write_text(json.dumps(best_metrics, indent=2))

    return comparison_results


def main() -> None:
    results = train_model()
    print("\n=== Model Comparison Results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()