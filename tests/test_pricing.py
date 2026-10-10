import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from streamlit.testing.v1 import AppTest

from src.data_processing import DATA_PATH, ROOT, load_data
from src.predict import FEATURES, PREPROCESSING_PATH, ModelLoadError, load_model
from src.pricing_engine import optimize_price


@pytest.fixture(scope="module")
def service():
    return load_model()


def test_loading_independent_of_working_directory(service, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(load_model().classes) == 2996
    assert len(load_data()) == 253009


def test_missing_model(tmp_path):
    with pytest.raises(ModelLoadError, match="Cannot load"):
        load_model(model_path=tmp_path / "missing.pkl")


def test_corrupt_model(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"invalid")
    with pytest.raises(ModelLoadError, match="checksum"):
        load_model(model_path=path)


@pytest.mark.parametrize("field,value", [("features", list(reversed(FEATURES))), ("scale", [0]*11)])
def test_invalid_preprocessing(tmp_path, field, value):
    meta = json.loads(PREPROCESSING_PATH.read_text())
    meta[field] = value
    path = tmp_path / "preprocessing.json"
    path.write_text(json.dumps(meta))
    with pytest.raises(ModelLoadError):
        load_model(preprocessing_path=path)


def test_changed_dataset(tmp_path):
    path = tmp_path / "changed.csv"
    path.write_text("changed")
    with pytest.raises(ModelLoadError, match="Dataset checksum"):
        load_model(data_path=path)


def test_notebook_prediction_parity(service):
    # Independently reconstruct notebook preprocessing from the preserved CSV.
    df = pd.read_csv(DATA_PATH, dtype={"StockCode": str})
    encoder = LabelEncoder()
    df["StockCode_enc"] = encoder.fit_transform(df.StockCode)
    df["elasticity"] = df.elasticity.fillna(0)
    df["price_sensitive"] = df.price_sensitive.astype(int)
    df["log_price"] = np.log1p(df.price)
    df["price_x_month"] = df.price * df.month
    df["price_x_dow"] = df.price * df.dow
    df = df.sort_values("date").reset_index(drop=True)
    df["week"] = pd.to_datetime(df.date).dt.isocalendar().week.astype(int)
    scaler = StandardScaler().fit(df[FEATURES].iloc[:int(len(df)*.8)])
    np.testing.assert_allclose(service.mean, scaler.mean_, rtol=0, atol=1e-12)
    np.testing.assert_allclose(service.scale, scaler.scale_, rtol=0, atol=1e-12)
    assert list(service.classes) == list(encoder.classes_)
    for i in [0, 202407, len(df)-1]:
        row = df.iloc[i]
        expected = service.model.booster_.predict(scaler.transform(df[FEATURES].iloc[[i]]), num_threads=2)
        actual = service.predict(row.StockCode, [row.price], row.date, row.n_txn, row.elasticity, row.price_sensitive)
        np.testing.assert_allclose(actual, np.maximum(expected, 0), rtol=1e-12)


@pytest.mark.parametrize("price", [0, -1, np.nan, np.inf])
def test_invalid_price(service, price):
    with pytest.raises(ValueError):
        service.predict("85123A", [price], "2010-11-01", 3)


def test_unknown_sku_and_invalid_transactions(service):
    with pytest.raises(ValueError, match="vocabulary"):
        service.predict("unknown", [1], "2010-11-01", 3)
    with pytest.raises(ValueError, match="Transaction"):
        service.predict("85123A", [1], "2010-11-01", 1.5)


def test_negative_and_nonfinite_predictions(service):
    with patch.object(service.model.booster_, "predict", return_value=np.array([-1.])):
        assert service.predict("85123A", [1], "2010-11-01", 3)[0] == 0
    with patch.object(service.model.booster_, "predict", return_value=np.array([np.nan])):
        with pytest.raises(RuntimeError):
            service.predict("85123A", [1], "2010-11-01", 3)


class LinearDemand:
    def predict(self, sku, prices, *args):
        return 10 - prices


def calculate(service=None, **kwargs):
    settings = dict(stock_code="test", date="2010-01-01", n_txn=1, elasticity=0,
                    price_sensitive=False, current_price=2, simulated_price=4,
                    min_price=1, max_price=9, points=9)
    settings.update(kwargs)
    return optimize_price(service or LinearDemand(), **settings)


def test_known_revenue_maximum():
    result = calculate()
    assert result.optimal_price == 5
    assert not result.boundary
    assert result.comparison.loc["Recommended", "revenue"] == 25
    assert result.uplift_pct == pytest.approx(56.25)
    np.testing.assert_allclose(result.curve.revenue, result.curve.price * result.curve.demand)


def test_bounds_and_baseline_inclusion():
    result = calculate(min_price=1, max_price=4, current_price=2.37)
    assert result.optimal_price == 4 and result.boundary
    assert 2.37 in result.curve.price.values
    with pytest.raises(ValueError):
        calculate(min_price=5, max_price=1)


def test_zero_revenue_tie():
    class ZeroDemand:
        def predict(self, sku, prices, *args):
            return np.zeros_like(prices)
    result = calculate(ZeroDemand())
    assert result.optimal_price == 1
    assert result.uplift_pct is None


def test_dashboard_interactions():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    assert not app.exception and not app.error
    assert len(app.metric) == 4
    app.sidebar.selectbox[0].select("10002").run()
    app.sidebar.number_input[2].set_value(1.25).run()
    assert not app.exception and not app.error
    app.sidebar.number_input[3].set_value(10.0).run()
    assert app.error and len(app.metric) == 0


def test_dashboard_model_failure():
    import streamlit as st
    st.cache_resource.clear()
    with patch("src.predict.load_model", side_effect=ModelLoadError("test missing model")):
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    assert app.error and not app.exception
    assert len(app.metric) == 0
    st.cache_resource.clear()
