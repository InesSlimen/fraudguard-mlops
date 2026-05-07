from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/samples/fraud_sample.parquet")

TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"


def load_ieee_cis_data() -> pd.DataFrame:
    if not TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing {TRANSACTION_FILE}. Place IEEE-CIS train_transaction.csv in data/raw/."
        )

    transactions = pd.read_csv(TRANSACTION_FILE)

    if IDENTITY_FILE.exists():
        identity = pd.read_csv(IDENTITY_FILE)
        data = transactions.merge(identity, on="TransactionID", how="left")
    else:
        data = transactions

    return data


def create_sample(n_rows: int = 50_000, seed: int = 42) -> pd.DataFrame:
    data = load_ieee_cis_data()

    if "isFraud" not in data.columns:
        raise ValueError("Expected target column 'isFraud' in training data.")

    sample_size = min(n_rows, len(data))
    sample = data.sample(n=sample_size, random_state=seed)

    return sample


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample = create_sample()
    sample.to_parquet(OUTPUT_PATH, index=False)

    print(f"Created real dataset sample: {OUTPUT_PATH}")
    print(f"Rows: {len(sample)}")
    print("Fraud distribution:")
    print(sample["isFraud"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
