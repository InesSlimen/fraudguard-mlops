import os

from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraudguard-baseline")


def select_best_run(metric_name: str = "pr_auc") -> str:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=20,
    )

    runs = [run for run in runs if metric_name in run.data.metrics]

    if not runs:
        raise RuntimeError(f"No runs found with metric: {metric_name}")

    return runs[0].info.run_id


if __name__ == "__main__":
    print(select_best_run())