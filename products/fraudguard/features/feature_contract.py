from dataclasses import dataclass

NUMERIC_FEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
]

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "isFraud"

REQUIRED_COLUMNS = FEATURES + [TARGET]


@dataclass(frozen=True)
class FeatureContract:
    numeric_features: list[str]
    categorical_features: list[str]
    target: str

    @property
    def features(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    @property
    def required_columns(self) -> list[str]:
        return self.features + [self.target]


def get_feature_contract() -> FeatureContract:
    return FeatureContract(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        target=TARGET,
    )


def validate_feature_columns(columns: list[str]) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]

    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")
