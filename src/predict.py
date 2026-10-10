"""Validated inference for the existing scaled LightGBM model."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing import DATA_PATH, ROOT

FEATURES = ["price", "log_price", "month", "dow", "week", "n_txn",
            "elasticity", "price_sensitive", "StockCode_enc", "price_x_month", "price_x_dow"]
MODEL_PATH = ROOT / "models/dynamic_pricing_model.pkl"
PREPROCESSING_PATH = ROOT / "models/preprocessing.json"


class ModelLoadError(RuntimeError):
    """The artifact cannot safely produce predictions."""


class PredictionService:
    def __init__(self, model, metadata):
        self.model = model
        self.mean = np.asarray(metadata["mean"], dtype=float)
        self.scale = np.asarray(metadata["scale"], dtype=float)
        self.classes = tuple(metadata["classes"])
        self.encoding = {sku: i for i, sku in enumerate(self.classes)}

    def predict(self, stock_code, prices, date, n_txn, elasticity=0.0, price_sensitive=False):
        """Daily units conditional on a fixed transaction count; retain full precision."""
        if str(stock_code) not in self.encoding:
            raise ValueError("This product was not in the training vocabulary.")
        prices = np.atleast_1d(np.asarray(prices, dtype=float))
        if prices.ndim != 1 or not prices.size or not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Prices must be a nonempty sequence of finite positive numbers.")
        if not np.isfinite(n_txn) or n_txn < 1 or int(n_txn) != n_txn:
            raise ValueError("Transaction count must be a positive integer.")
        elasticity = 0.0 if elasticity is None or pd.isna(elasticity) else float(elasticity)
        if not np.isfinite(elasticity) or price_sensitive not in (True, False, 0, 1):
            raise ValueError("Invalid elasticity or price-sensitive flag.")
        day = pd.Timestamp(date)
        if pd.isna(day):
            raise ValueError("A valid scenario date is required.")
        row = pd.DataFrame({
            "price": prices, "log_price": np.log1p(prices), "month": day.month,
            "dow": day.dayofweek, "week": day.isocalendar().week, "n_txn": n_txn,
            "elasticity": elasticity, "price_sensitive": int(price_sensitive),
            "StockCode_enc": self.encoding[str(stock_code)],
            "price_x_month": prices * day.month, "price_x_dow": prices * day.dayofweek,
        })
        scaled = (row[FEATURES].to_numpy() - self.mean) / self.scale
        if not np.isfinite(scaled).all():
            raise ValueError("Scenario features exceed the supported numeric range.")
        # Native Booster accepts the unnamed scaled matrix used during training.
        pred = np.asarray(self.model.booster_.predict(scaled, num_threads=2), dtype=float)
        if pred.shape != prices.shape or not np.isfinite(pred).all():
            raise RuntimeError("The model returned invalid predictions.")
        return np.maximum(pred, 0.0)


def load_model(model_path=MODEL_PATH, preprocessing_path=PREPROCESSING_PATH, data_path=DATA_PATH):
    """Load trusted local artifacts, failing closed on mismatched preprocessing."""
    try:
        raw = Path(model_path).read_bytes()
        meta = json.loads(Path(preprocessing_path).read_text())
        if hashlib.sha256(raw).hexdigest() != meta["model_sha256"]:
            raise ValueError("Model checksum does not match verified preprocessing.")
        if hashlib.sha256(Path(data_path).read_bytes()).hexdigest() != meta["data_sha256"]:
            raise ValueError("Dataset checksum does not match verified preprocessing.")
        bundle = pickle.loads(raw)
        if meta["schema_version"] != 1 or bundle["features"] != FEATURES or meta["features"] != FEATURES:
            raise ValueError("Unsupported feature schema or order.")
        model = bundle["model"]
        from lightgbm import LGBMRegressor
        if not isinstance(model, LGBMRegressor) or model.booster_.num_feature() != len(FEATURES):
            raise ValueError("Expected a fitted LightGBM regressor with 11 features.")
        service = PredictionService(model, meta)
        if (service.mean.shape != (11,) or service.scale.shape != (11,)
                or not np.isfinite(service.mean).all() or not np.isfinite(service.scale).all()
                or (service.scale <= 0).any() or not service.classes
                or list(service.classes) != sorted(set(service.classes))):
            raise ValueError("Invalid preprocessing parameters.")
        service.predict(service.classes[0], [1.0], "2010-11-01", 1)
        return service
    except Exception as exc:
        raise ModelLoadError(f"Cannot load verified pricing model: {exc}") from exc
