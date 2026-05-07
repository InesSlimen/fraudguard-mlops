import json

from ci.quality_gates_import import import_check_metrics

from products.fraudguard.training.train import train_model
from scripts.create_synthetic_sample import create_synthetic_sample


def test_train_model_creates_metrics(tmp_path, monkeypatch):
    data_path = tmp_path / "fraud_sample.parquet"
    df = create_synthetic_sample(n_rows=500, seed=42)
    df.to_parquet(data_path, index=False)

    metrics = train_model(data_path)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_metrics_quality_gate_passes(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "roc_auc": 0.75,
                "pr_auc": 0.10,
                "precision": 0.20,
                "recall": 0.80,
                "f1": 0.32,
                "threshold": 0.5,
            }
        )
    )

    check_metrics = import_check_metrics()
    report = check_metrics(metrics_path)

    assert report["status"] == "passed"
