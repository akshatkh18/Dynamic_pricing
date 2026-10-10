# Dynamic Pricing — Revenue Studio

A Streamlit dashboard for exploring daily product demand and revenue with the existing LightGBM model. Select a SKU, set a date and assumed daily transaction-line count, compare a baseline and simulated price, and search a bounded range for the highest modeled revenue.

## Local setup

Use Python **3.13** (verified locally with 3.13.5). From the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m streamlit run app/streamlit_app.py
```

For application-only installation, use `requirements.txt`. No API keys, external services, raw Excel loading, or training jobs are needed to run the dashboard. Paths resolve from the source files rather than the current working directory.

## What is actually in this repository

- `app/streamlit_app.py`: dashboard with SKU selection, scenario inputs, demand/revenue curves, comparison table, bounded recommendation and historical context.
- `src/predict.py`: verified model loading, feature construction, preprocessing and batched inference.
- `src/pricing_engine.py`: bounded grid search maximizing price × predicted units.
- `src/data_processing.py`: portable processed-data loader.
- `models/dynamic_pricing_model.pkl`: original trained model, preserved unchanged.
- `models/preprocessing.json`: recovered preprocessing, vocabulary, checksums and verification provenance.
- `scripts/recover_preprocessing.py`: reproducible preprocessing recovery, without model retraining.
- `data/processed/dynamic_pricing_model_ready.csv`: 253,009 product-day observations, 2,996 SKUs, 2009-12-01 through 2010-12-09.
- `data/processed/elasticity_by_sku.csv`: historical significant elasticity estimates.
- `data/raw/online_retail_II.xlsx`: preserved source dataset.
- `notebooks/`: original exploration and model comparison notebooks, preserved unchanged.
- `tests/test_pricing.py`: artifact, prediction, optimizer and dashboard tests.

`api/`, `run.py`, `src/train.py` and `config/config.yaml` are legacy placeholders, not implemented interfaces. Launch Streamlit directly using the command above. The notebooks require additional exploratory dependencies (matplotlib, seaborn, openpyxl, joblib, xgboost, catboost and a notebook runtime); these are intentionally excluded from serving dependencies. Their original relative input/output paths must be adjusted before rerunning them. No automated training command is provided.

## Artifact audit and compatibility

The notebook saves a four-key bundle (`model`, `scaler`, `features`, `le`) as `dynamic_pricing_lgbm.pkl`. The actual committed `dynamic_pricing_model.pkl` is a pickle dictionary containing **only `model` and `features`**. It holds a fitted LightGBM LGBMRegressor with 200 trees, maximum depth 5 and learning rate 0.05, trained on scaled features.

Feature order:

```text
price, log_price, month, dow, week, n_txn, elasticity,
price_sensitive, StockCode_enc, price_x_month, price_x_dow
```

Recovery reproduces the notebook: fill missing elasticity with zero; encode string SKUs in LabelEncoder's sorted order; use log1p(price), price × month and price × weekday; sort string dates; derive ISO week; fit StandardScaler on the first 80% (202,407 rows). All 11 reconstructed scaled training minima/maxima match the saved Booster's feature ranges exactly in the audited environment. The original scaler was absent, so this is a verified reconstruction, not retrieval of an original scaler. Runtime uses the stored mean/scale and vocabulary; it never fits on incoming scenarios.

To reproduce the sidecar (only when intentionally verifying the original artifacts):

```bash
python -m scripts.recover_preprocessing
```

Recovery refuses to write if training ranges disagree. Model and dataset SHA-256 checks prevent accidental use of a different artifact with this preprocessing. Only load trusted repository pickle files. Missing, corrupt or incompatible artifacts stop the dashboard without substitute predictions. Unknown SKUs are rejected instead of inventing an encoding. Negative demand is clipped to zero as in the notebook; rounding is deferred to display so revenue calculations retain precision.

The saved artifact's original library versions are not recorded. The pinned requirements are the versions used for successful local compatibility tests, not claimed original training versions.

## Interpretation and limitations

The audit's last-20%-of-rows evaluation produced MAE **10.0534 units** and R² **0.5169**, using unrounded model outputs. These are descriptive reconstruction checks, not production guarantees:

- Elasticity was estimated using the full dataset, including the evaluation period. The row-based temporal split can also divide one date between train and test.
- `n_txn` is same-day transaction-line count, not unique invoices. It is unknown ahead of time and correlated with sales. The dashboard holds the user's assumption fixed across candidate prices.
- Records cover observed sales days, not a complete calendar with zero-sales days; the model is not an unconditional daily demand forecast.
- The data is from 2009–2010. The model has seasonal features but no trend or current-market information.
- Historical price associations are not causal effects. Predictions can be flat or non-monotonic. No confidence intervals, inventory, costs, competition, or profit optimization are included.

All comparisons use model predictions under identical assumptions. The baseline defaults to the latest historical mean price rounded to pennies. Search evaluates up to 101 equally spaced prices rounded to pennies, plus baseline and simulation prices if in bounds. Ties select the lower price; zero baseline revenue gives undefined percentage uplift. An optimum at a boundary is flagged. Recommendations are the best **tested** prices inside user bounds; validate any real price change using current data and controlled experiments.

## Streamlit Community Cloud deployment

After reviewing and verifying the changes locally:

1. Commit and push the implementation, `requirements.txt`, `.streamlit/config.toml`, original model, `models/preprocessing.json` and both processed CSVs to the branch you intend to deploy. The current local branch is `feature/streamlit-app`. Do not commit virtual environments, secrets, caches or local runtime dependencies.
2. Sign in at https://share.streamlit.io and connect the GitHub account with access to `akshatkh18/Dynamic_pricing`.
3. Choose **Create app**, then deploy from the repository.
4. Set repository to **akshatkh18/Dynamic_pricing**, branch to your verified pushed branch (for this checkout, **feature/streamlit-app**), and main file path to **app/streamlit_app.py**.
5. In **Advanced settings**, choose **Python 3.13**. No secrets are needed. Deploy.
6. Confirm the app shows four recommendation metrics, change SKU and simulated price, and inspect the comparison. Check build logs if dependencies fail. The repository-root `requirements.txt` supplies serving dependencies; the original model/data must be present in that branch.

Official instructions: [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

No push or cloud deployment is performed by these setup or verification commands.

## Verification performed

On Python 3.13.5 with the pinned serving dependencies: **18 tests passed**, covering notebook preprocessing/prediction parity, portable paths, missing/corrupt/mismatched artifacts, invalid inputs, nonfinite model outputs, nonnegative demand, revenue arithmetic, known optimal prices, bounded search, ties, zero-baseline revenue, Streamlit widget interactions and failure-state rendering. Streamlit also started as a server and returned HTTP **200 `ok`** from `/_stcore/health`. `git diff --check` passed. The original model, datasets and notebooks are unchanged.

The local Streamlit/Altair combination emits one upstream theme deprecation warning during chart tests; it does not prevent rendering or test success. Linux/Community Cloud deployment itself has not been executed. Windows sandbox restrictions required running temporary-directory tests and the localhost health check with permission outside the sandbox.
