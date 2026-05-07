from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path("data/samples/fraud_sample.parquet")


def create_synthetic_sample(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    fraud_probability = 0.035

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n_rows + 1),
            "TransactionAmt": rng.lognormal(mean=4.0, sigma=1.0, size=n_rows).round(2),
            "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=n_rows),
            "card1": rng.integers(1000, 20000, size=n_rows),
            "card2": rng.choice([111, 150, 321, 555, np.nan], size=n_rows),
            "card3": rng.choice([150, 185, 106, np.nan], size=n_rows),
            "card4": rng.choice(
                ["visa", "mastercard", "american express", "discover", None], size=n_rows
            ),
            "card5": rng.choice([102, 117, 126, 195, 226, np.nan], size=n_rows),
            "card6": rng.choice(["credit", "debit", None], size=n_rows),
            "addr1": rng.choice([204, 299, 330, 441, np.nan], size=n_rows),
            "addr2": rng.choice([87, 96, np.nan], size=n_rows),
            "P_emaildomain": rng.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None],
                size=n_rows,
            ),
            "R_emaildomain": rng.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None],
                size=n_rows,
            ),
            "DeviceType": rng.choice(["desktop", "mobile", None], size=n_rows),
            "DeviceInfo": rng.choice(["Windows", "iOS", "Android", "MacOS", None], size=n_rows),
        }
    )

    risk_score = (
        (df["TransactionAmt"] > df["TransactionAmt"].quantile(0.9)).astype(float) * 0.20
        + (df["ProductCD"].isin(["C", "S"])).astype(float) * 0.10
        + (df["card6"] == "credit").astype(float) * 0.05
        + rng.random(n_rows) * fraud_probability
    )

    threshold = np.quantile(risk_score, 0.965)
    df["isFraud"] = (risk_score >= threshold).astype(int)

    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = create_synthetic_sample()
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Created synthetic sample: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print("Fraud distribution:")
    print(df["isFraud"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
