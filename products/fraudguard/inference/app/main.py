import os

from fastapi import FastAPI, HTTPException

from products.fraudguard.inference.app.features import request_to_dataframe
from products.fraudguard.inference.app.model_loader import ModelLoader
from products.fraudguard.inference.app.schemas import PredictionRequest, PredictionResponse

MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "local")

app = FastAPI(
    title="FraudGuard Inference API",
    version="0.1.0",
    description="Real-time fraud detection inference service.",
)

model_loader = ModelLoader()


@app.on_event("startup")
def startup() -> None:
    model_loader.load()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    return {"ready": model_loader.is_ready()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model is not ready.")

    dataframe = request_to_dataframe(request)
    probability = model_loader.predict_probability(dataframe)
    decision = "review" if probability >= MODEL_THRESHOLD else "approve"

    return PredictionResponse(
        fraud_probability=probability,
        decision=decision,
        threshold=MODEL_THRESHOLD,
        model_version=MODEL_VERSION,
    )