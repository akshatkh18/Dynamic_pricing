"""Repository-relative access to the preserved daily product dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dynamic_pricing_model_ready.csv"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, dtype={"StockCode": str}, parse_dates=["date"])
    required = {"date", "StockCode", "price", "demand", "revenue", "n_txn",
                "month", "dow", "elasticity", "price_sensitive"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError("The processed dataset is empty or missing required columns.")
    numeric = df[["price", "demand", "revenue", "n_txn", "month", "dow"]]
    if not np.isfinite(numeric.to_numpy()).all() or df.date.isna().any():
        raise ValueError("The processed dataset contains invalid values.")
    if (df.price <= 0).any() or (df.demand < 0).any() or (df.n_txn < 1).any():
        raise ValueError("Prices, demand, or transaction counts are invalid.")
    return df
