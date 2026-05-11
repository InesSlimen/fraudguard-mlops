from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    TransactionAmt: float
    ProductCD: str | None = None
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(ge=0.0, le=1.0)
    decision: str
    threshold: float
    model_version: str