import pandas as pd
import pytest

from products.fraudguard.features.build_features import build_feature_dataset
from products.fraudguard.features.feature_contract import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    validate_feature_columns,
)
from products.fraudguard.features.preprocessors import build_preprocessor
from scripts.create_synthetic_sample import create_synthetic_sample


def test_feature_contract_contains_expected_columns():
    assert "TransactionAmt" in NUMERIC_FEATURES
    assert "ProductCD" in CATEGORICAL_FEATURES
    assert TARGET == "isFraud"
    assert set(FEATURES) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_validate_feature_columns_passes_for_valid_data():
    df = create_synthetic_sample(n_rows=100, seed=42)

    validate_feature_columns(list(df.columns))


def test_validate_feature_columns_fails_for_missing_columns():
    df = create_synthetic_sample(n_rows=100, seed=42)
    df = df.drop(columns=["TransactionAmt"])

    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_feature_columns(list(df.columns))


def test_build_feature_dataset_returns_x_and_y(tmp_path):
    df = create_synthetic_sample(n_rows=100, seed=42)
    data_path = tmp_path / "sample.parquet"
    df.to_parquet(data_path, index=False)

    X, y = build_feature_dataset(data_path)

    assert list(X.columns) == FEATURES
    assert y.name == TARGET
    assert len(X) == 100
    assert len(y) == 100


def test_preprocessor_handles_missing_values_and_unknown_categories():
    train_df = create_synthetic_sample(n_rows=200, seed=42)
    X_train = train_df[FEATURES]

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    inference_row = pd.DataFrame(
        [
            {
                "TransactionAmt": 199.99,
                "ProductCD": "UNKNOWN_PRODUCT",
                "card1": 12345,
                "card2": None,
                "card3": 150,
                "card4": "unknown_card",
                "card5": None,
                "card6": "unknown_type",
                "addr1": None,
                "addr2": 87,
                "P_emaildomain": "newdomain.com",
                "R_emaildomain": None,
                "DeviceType": "console",
                "DeviceInfo": "UnknownDevice",
            }
        ]
    )

    transformed = preprocessor.transform(inference_row)

    assert transformed.shape[0] == 1