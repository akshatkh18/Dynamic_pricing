# AI-Based Dynamic Pricing System

A production-style ML system that predicts demand and recommends the optimal price to maximize revenue.

---

## 🏗️ Architecture

```
raw data → DataIngestion → DataTransformation → ModelTrainer → best_model.pkl
                                                                      ↓
                                              PriceOptimizer (simulate prices)
                                                                      ↓
                                              Streamlit Dashboard / FastAPI
```

---

## 📁 Project Structure

```
dynamic-pricing/
├── config/
│   └── config.yaml              ← centralized configuration
├── data/
│   ├── raw/                     ← original CSV goes here
│   └── processed/               ← cleaned, feature-engineered data
├── models/                      ← saved model + preprocessor
├── notebooks/                   ← EDA and experimentation
├── src/
│   ├── components/
│   │   ├── data_ingestion.py    ← load + validate raw data
│   │   ├── data_transformation.py ← feature engineering + preprocessing
│   │   └── model_trainer.py     ← train, compare, save best model
│   ├── pipeline/
│   │   ├── train_pipeline.py    ← orchestrates training end-to-end
│   │   └── predict_pipeline.py  ← handles inference
│   ├── pricing/
│   │   ├── optimizer.py         ← core pricing engine (simulate → optimize)
│   │   └── simulator.py         ← compare current vs optimal price
│   └── utils/
│       ├── logger.py
│       ├── exception.py
│       └── helpers.py           ← shared feature engineering + I/O
├── api/
│   └── main.py                  ← FastAPI REST API
├── app/
│   └── streamlit_app.py         ← Streamlit dashboard
├── run.py                       ← single entry point
├── setup.py
└── requirements.txt
```

---

## ⚡ Quickstart

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your data
Place your CSV at:
```
data/raw/online_retail.csv
```
Required columns: `price`, `demand`, `date`

### 4. Run training pipeline
```bash
python run.py
```

### 5. Start the dashboard
```bash
python run.py --app
```

### 6. Start the API
```bash
python run.py --api
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict demand for a given price |
| POST | `/optimize` | Find optimal price for max revenue |
| POST | `/simulate` | Compare current price vs optimal |

### Example: Optimize price
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "month": 6,
    "dow": 2,
    "week": 24,
    "n_txn": 10,
    "elasticity": -1.2,
    "price_sensitive": 0,
    "price_min": 1.0,
    "price_max": 20.0
  }'
```

---

## 🧠 Key Design Decisions

| Decision | Reason |
|----------|--------|
| Time-based train/test split | Prevents data leakage from future |
| Removed StockCode from features | High cardinality → overfitting |
| MAE as primary metric | Business-relevant, robust to outliers |
| Price simulation (not analytic) | Tree models have no closed-form derivative |
| Shared `engineer_features()` | Guarantees identical transform at train + inference |

---

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| `price` | Core input — what we're optimizing |
| `log_price` | Captures non-linear price sensitivity |
| `month` | Seasonal demand pattern |
| `dow` | Day-of-week demand pattern |
| `week` | Weekly seasonality |
| `n_txn` | Transaction volume signal |
| `elasticity` | Demand sensitivity to price |
| `price_sensitive` | Binary customer segment flag |
| `price_x_month` | Price × seasonal interaction |
| `price_x_dow` | Price × weekday interaction |

---

## 🔁 System Flow

```
price_min → price_max
     ↓
for each candidate price:
    features = base_features + price + derived
    demand   = model.predict(features)
    revenue  = price × demand
     ↓
return price with max revenue
```

---

## Author
Akshat Khandelwal — akshatkhandelwal1805@gmail.com