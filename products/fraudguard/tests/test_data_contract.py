import pandas as pd
import pandera as pa
import pytest

from contracts.data_contract import get_data_schema


def test_valid_data_passes():
    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionAmt": [100.0],
            "ProductCD": ["W"],
            "card1": [1000],
            "card2": [150.0],
            "card3": [150.0],
            "card4": ["visa"],
            "card5": [100.0],
            "card6": ["credit"],
            "addr1": [100.0],
            "addr2": [87.0],
            "P_emaildomain": ["gmail.com"],
            "R_emaildomain": ["gmail.com"],
            "DeviceType": ["desktop"],
            "DeviceInfo": ["Windows"],
            "isFraud": [0],
        }
    )

    schema = get_data_schema()
    validated = schema.validate(df)

    assert validated is not None


def test_invalid_data_fails():
    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionAmt": [-100.0],  # invalid
            "isFraud": [2],  # invalid
        }
    )

    schema = get_data_schema()

    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)
