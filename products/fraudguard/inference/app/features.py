import pandas as pd

from products.fraudguard.features.feature_contract import FEATURES
from products.fraudguard.inference.app.schemas import PredictionRequest


def request_to_dataframe(request: PredictionRequest) -> pd.DataFrame:
    payload = request.model_dump()
    row = {feature: payload.get(feature) for feature in FEATURES}
    return pd.DataFrame([row])