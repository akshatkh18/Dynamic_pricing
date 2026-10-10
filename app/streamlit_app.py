"""Streamlit entry point: python -m streamlit run app/streamlit_app.py."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.data_processing import load_data
from src.predict import load_model
from src.pricing_engine import optimize_price

st.set_page_config(page_title="Dynamic Pricing | Revenue Studio", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2.5rem; max-width: 1400px;}
[data-testid="stMetric"] {background: #f0f5fa; border: 1px solid #dce6ef;
 border-radius: 12px; padding: 18px;}
[data-testid="stMetricLabel"] {color: #465b70;}
[data-testid="stMetricValue"] {color: #122d46;}
</style>""", unsafe_allow_html=True)
st.caption("REVENUE STUDIO / DYNAMIC PRICING")
st.title("Find the price. Understand the trade-off.")
st.write("Explore daily demand and revenue scenarios with the repository’s trained LightGBM model.")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_data():
    return load_data()


try:
    service = get_model()
    data = get_data()
except Exception as exc:
    st.error("Pricing is unavailable. The model or its verified dataset could not be loaded.")
    st.code(str(exc))
    st.info("Restore the committed model, preprocessing.json and processed CSV, then install requirements.txt and restart.")
    st.stop()

products = sorted(set(data.StockCode) & set(service.classes))
if not products:
    st.error("No products in the dataset match the trained model.")
    st.stop()
with st.sidebar:
    st.header("Scenario controls")
    sku = st.selectbox("Product / SKU", products, index=products.index("85123A") if "85123A" in products else 0)
    history = data[data.StockCode == sku].sort_values("date")
    latest = history.iloc[-1]
    st.caption(f"{len(history):,} observed sales days · {history.date.min():%d %b %Y} – {history.date.max():%d %b %Y}")
    date = st.date_input("Scenario date", value=latest.date.date(), key=f"date_{sku}")
    n_txn = st.number_input("Assumed daily transaction lines", min_value=1, max_value=10000,
                            value=int(latest.n_txn), step=1, key=f"txn_{sku}")
    st.caption("Held fixed as prices change. This is an assumption, not a forecast of customer traffic.")
    current = st.number_input("Baseline unit price (£)", min_value=0.01, value=round(float(latest.price), 2),
                              step=0.05, key=f"base_{sku}")
    simulated = st.number_input("Simulated unit price (£)", min_value=0.01, value=round(current * 1.1, 2),
                                step=0.05, key=f"sim_{sku}")
    st.subheader("Recommendation bounds")
    low_default = max(0.01, round(max(float(history.price.min()), current * .8), 2))
    high_default = max(low_default, round(min(float(history.price.max()), current * 1.2), 2))
    low = st.number_input("Minimum price (£)", min_value=0.01, value=low_default, step=0.05, key=f"low_{sku}")
    high = st.number_input("Maximum price (£)", min_value=0.01, value=high_default, step=0.05, key=f"high_{sku}")
    st.caption("Searches up to 101 rounded price points plus in-range baseline and simulation prices.")

st.info("Historical scenario analysis, not a live forecast. Recommendations maximize modeled revenue within your bounds; they do not establish a causal price effect or guaranteed uplift.")
if min(low, current, simulated) < history.price.min() or max(high, current, simulated) > history.price.max():
    st.warning(f"Some prices are outside this SKU’s observed range (£{history.price.min():.2f}–£{history.price.max():.2f}). Predictions there are extrapolations.")
if pd.Timestamp(date) < data.date.min() or pd.Timestamp(date) > data.date.max():
    st.warning("The scenario date is outside the 2009–2010 dataset. The model has seasonal features but no trend or current-market information.")
try:
    result = optimize_price(service, stock_code=sku, date=date, n_txn=n_txn,
                            elasticity=latest.elasticity, price_sensitive=latest.price_sensitive,
                            current_price=current, simulated_price=simulated, min_price=low, max_price=high)
except Exception as exc:
    st.error(f"Cannot calculate this scenario: {exc}")
    st.stop()

best = result.comparison.loc["Recommended"]
simulation = result.comparison.loc["Simulation"]
st.subheader(f"SKU {sku} · Price opportunity")
a, b, c, d = st.columns(4)
a.metric("Recommended price", f"£{result.optimal_price:.2f}")
b.metric("Recommended daily units", f"{best.demand:,.1f}")
c.metric("Recommended daily revenue", f"£{best.revenue:,.2f}")
d.metric("Modeled revenue vs baseline", "N/A" if result.uplift_pct is None else f"{result.uplift_pct:+.1f}%")
if result.boundary:
    st.warning("The best tested price is at a search boundary. This is a bounded result, not evidence of a global optimum.")
if best.revenue == 0:
    st.warning("All tested revenues are zero. There is no modeled revenue advantage; the lowest tested price is shown as a tie-break.")

left, right = st.columns(2)
with left:
    st.subheader("Demand response")
    st.caption("Predicted units per day, holding the scenario assumptions fixed")
    st.line_chart(result.curve.set_index("price")[["demand"]], x_label="Unit price (£)", y_label="Daily units", color="#168a8a")
with right:
    st.subheader("Revenue opportunity")
    st.caption("Unit price × predicted daily demand")
    st.line_chart(result.curve.set_index("price")[["revenue"]], x_label="Unit price (£)", y_label="Daily revenue (£)", color="#4169c1")
st.subheader("Compare your scenarios")
st.dataframe(result.comparison.rename(columns={"price": "Unit price (£)", "demand": "Predicted daily units", "revenue": "Predicted daily revenue (£)"})
             .style.format({"Unit price (£)": "£{:.2f}", "Predicted daily units": "{:.1f}", "Predicted daily revenue (£)": "£{:.2f}"}), use_container_width=True)
st.caption("All three rows use model predictions under the same assumptions. Baseline defaults to the latest historical mean unit price, rounded to pennies. Display rounding is not used in revenue calculations.")

with st.expander("Historical context & model limitations"):
    st.write(f"Observed price range: £{history.price.min():.2f}–£{history.price.max():.2f}. Latest observed daily demand: {latest.demand:,.0f} units on {latest.date:%d %b %Y}.")
    st.line_chart(history.set_index("date")[["demand"]], y_label="Observed units on recorded sales days", color="#168a8a")
    st.write("The dataset contains sales days only, not a complete calendar with zero-sales days. n_txn counts transaction lines, not unique invoices. Elasticity was estimated using the full historical dataset, including the evaluation period; holdout scores are therefore descriptive and may be optimistic.")
    st.write("The original model is unchanged. Its missing scaler and SKU encoding were reconstructed from the notebook and verified against all 11 embedded training feature ranges. Missing elasticity is filled with zero. Negative predicted units are clipped to zero; no synthetic fallback predictions are used.")
    st.write("The search optimizes revenue, not profit. Costs, inventory, competitor prices, causal elasticity and prediction intervals are not modeled. Tree predictions may be flat or non-monotonic; validate any price change with current data and a controlled experiment.")
st.caption("LightGBM · 11 verified features · 2,996 known products · GBP · Preserved model and datasets")
