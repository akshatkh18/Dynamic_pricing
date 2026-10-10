"""Recover the omitted notebook preprocessing; never refit the demand model.

Run from the repository root: python -m scripts.recover_preprocessing
Only the repository's trusted pickle should be loaded.
"""
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data_processing import DATA_PATH, ROOT


def main():
    model_path = ROOT / "models/dynamic_pricing_model.pkl"
    bundle = pickle.loads(model_path.read_bytes())
    df = pd.read_csv(DATA_PATH, dtype={"StockCode": str})
    df["elasticity"] = df.elasticity.fillna(0)
    df["price_sensitive"] = df.price_sensitive.astype(int)
    encoder = LabelEncoder()
    df["StockCode_enc"] = encoder.fit_transform(df.StockCode)
    df["log_price"] = np.log1p(df.price)
    df["price_x_month"] = df.price * df.month
    df["price_x_dow"] = df.price * df.dow
    # Match the notebook's string-date sort, including its default tie ordering.
    df = df.sort_values("date").reset_index(drop=True)
    df["week"] = pd.to_datetime(df.date).dt.isocalendar().week.astype(int)
    split = int(len(df) * 0.8)
    scaler = StandardScaler().fit(df[bundle["features"]].iloc[:split])
    scaled = scaler.transform(df[bundle["features"]].iloc[:split])
    info = bundle["model"].booster_.dump_model()["feature_infos"]
    expected = np.array([[v["min_value"], v["max_value"]] for v in info.values()])
    actual = np.array([scaled.min(axis=0), scaled.max(axis=0)]).T
    if not np.allclose(expected, actual, rtol=0, atol=1e-12):
        raise ValueError("Recovered preprocessing does not match the model's training ranges.")
    result = {
        "schema_version": 1,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "features": bundle["features"], "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "classes": encoder.classes_.tolist(),
        "training_rows": split, "training_ranges": expected.tolist(),
        "recovery": "Notebook preprocessing; all 11 scaled training ranges verified at atol=1e-12",
    }
    (ROOT / "models/preprocessing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified preprocessing for {split:,} training rows and {len(encoder.classes_):,} SKUs.")


if __name__ == "__main__":
    main()
