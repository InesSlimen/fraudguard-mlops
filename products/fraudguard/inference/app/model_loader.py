from pathlib import Path
from typing import Any

import joblib

DEFAULT_MODEL_PATH = Path("artifacts/fraud_model.joblib")


class ModelLoader:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.model: Any | None = None

    def load(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}. "
                "Run `make train` before starting the API."
            )

        self.model = joblib.load(self.model_path)
        return self.model

    def is_ready(self) -> bool:
        return self.model is not None

    def predict_probability(self, dataframe) -> float:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        return float(self.model.predict_proba(dataframe)[0, 1])