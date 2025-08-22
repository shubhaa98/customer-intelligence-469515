# Customer Intelligence: Forecasting, Churn, Segmentation & Sentiment

A complete, notebook-driven project for retail **customer intelligence**:
- **Forecasting** product,week,region and category level sales forcasts
- **Churn modeling** (classification with ROC-AUC/F1 reporting)
- **Customer segmentation** (clustering + interpretable labels)
- **Sentiment analysis** (NLTK-based polarity; optional topic modeling)
- Solid **EDA** to diagnose variance, sparsity, and seasonality

The repo is structured around reproducible notebooks and a lightweight container for deployment.

---

## 📦 Repository Structure

```
.
├── data/                         # (optional) put your local CSVs here (not committed by default)
│   ├── cleaned_customer_data.csv
│   └── customer_intelligence_dataset.csv
├── eda.ipynb                     # High-level EDA across sales, customers, products
├── regression eda.ipynb          # EDA specific to regression/forecasting targets
├── train_forecasting.ipynb       # Monthly demand forecasting (lag/rolling + models)
├── train_churn.ipynb             # Churn modeling (LogReg / RF / GBDT / XGB if available)
├── train_segmentation.ipynb      # Customer clustering + labels
├── train_sentiment_analysis.ipynb# Sentiment pipeline (NLTK VADER etc.)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container for serving / batching
└── README.md                     # You are here
```

> **Tip**: If you don’t want to check large/private CSVs into Git, keep them in `data/` and add `/data/` to `.gitignore`. Provide collaborators with a download link or an S3/Drive sync script.

---

## 🧰 Data

This project expects two primary CSVs (column names may vary slightly):
- `cleaned_customer_data.csv`
- `customer_intelligence_dataset.csv`

At minimum, the following fields are commonly used across tasks:
- **Transaction-level**: `sale_id`, `customer_id`, `product_id`, `category`, `region`, `price`, `quantity`, `sale_date`, `total_value`
- **Customer-level**: `age`, `gender`, `segment`, `tenure_months`, `churn`
- **Text**: `feedback_text`, with derived `sentiment`
- **Time-derivatives**: `year`, `month`, `quarter`

> Place the CSVs in `data/` and update notebook paths if needed.

---

## 🔎 EDA (Exploratory Data Analysis)

- Inspect **target distributions** (quantity, churn) for skew/imbalance
- Study **time series** (trend, seasonality, variance) via monthly groupbys
- Evaluate **group variance** (e.g., category×region Coefficient of Variation)
- Identify **sparsity** (missing months/segments)
- Run **correlation** and simple **lag baselines** for forecasting sanity checks

Files: `eda.ipynb`, `regression eda.ipynb`

---

## 📈 Forecasting (Monthly Quantity)

Notebook: `train_forecasting.ipynb`

**Approach**
- Aggregate to **product_id × month** (optionally category/region granularity)
- Engineer **lags** (t–1, t–2, t–3), **rolling means/sums**
- Add **calendar features** (month/quarter + cyclical encodings)
- Train a few tabular models (Linear Regression, Random Forest, Gradient Boosting, XGBoost if installed)
- Evaluate with **R²** and **RMSE** on a **time-based split** (e.g., last 6 months)

**Notes**
- Forecasting individual transactions is noisy; monthly aggregation is recommended
- Where seasonality is present, you can optionally try **Prophet** or **SARIMA**

---

## 🔔 Churn Modeling

Notebook: `train_churn.ipynb`

**Approach**
- Drop identifiers, raw dates, free text, and target-adjacent columns to avoid leakage
- One-Hot encode categoricals; scale numeric features
- Train **Logistic Regression**, **Random Forest**, **Gradient Boosting** (and **XGBoost** if available)
- Compare **ROC-AUC**, **F1**, **Precision/Recall**, **Accuracy**
- Plot **ROC** and **confusion matrix**
- Save the best pipeline (preprocessing + model) as a pickle for reuse

**Outputs**
- `best_churn_model.pkl` (saved by the notebook)
- Metrics tables and diagnostic plots

---

## 👥 Customer Segmentation

Notebook: `train_segmentation.ipynb`

**Approach**
- Construct customer-level features: **tenure**, **total_spend**, **total_quantity**, **purchase_frequency**, **avg_order_value**, etc.
- Standardize features and cluster (e.g., **KMeans**)
- Label clusters with short, actionable names (e.g., *“Frequent Spenders (Short‑Term)”, “Loyal Mid‑Tenure”, “Dormant High‑Tenure”*)
- Validate with **silhouette** and by business interpretability

**Outputs**
- Cluster assignments + profile report
- (Optional) per‑cluster marketing playbook

---

## 💬 Sentiment Analysis

Notebook: `train_sentiment_analysis.ipynb`

**Approach**
- Use **NLTK VADER** for quick polarity scores (Positive/Negative/Neutral)
- (Optional) Topic modeling via **LDA** for themes
- Map text to a simple **sentiment class** and aggregate to customer/product levels

**Outputs**
- Sentiment labels per feedback
- Aggregated sentiment features for downstream models

---

## 🚀 Quickstart (Local)

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Start Jupyter and run notebooks
jupyter notebook
# or
jupyter lab
```

> Key dependencies are listed in `requirements.txt` (sklearn, pandas, numpy, matplotlib/seaborn, Flask/gunicorn for serving, Prophet/statsmodels for TS, imbalanced-learn, nltk, wordcloud, SHAP/LIME, etc.).

---

## 🐳 Docker (Optional)

A `Dockerfile` is included to containerize training or a lightweight API if you add an `app.py`:
```bash
# Build
docker build -t customer-intel:latest .

# Run (example; adjust ports/entrypoint as needed)
docker run --rm -p 8080:8080 customer-intel:latest
```

> Ensure the `CMD`/`ENTRYPOINT` in `Dockerfile` matches your actual app (e.g., `gunicorn --bind 0.0.0.0:8080 app:app`).

---

## 🧪 Reproducibility

- Pin dependencies via `requirements.txt`
- Keep raw data out of Git if large/private; record download/source steps
- Use **time-based splits** for forecasting; **stratified splits** for churn
- Log model configs, seeds, and evaluation windows

---

## 📁 Data & Paths

- Default notebooks assume CSVs live under `data/`
- If your files are elsewhere, adjust paths (e.g., `../data/...` → `data/...`)
- For deployments, mount or pass data paths through env/config

---

## 🔒 Notes on Privacy & Governance

- Remove PII from datasets before sharing
- Keep model artifacts and data under appropriate access controls
- Document data lineage and consent where applicable

---

## 📝 License

Add your preferred license (e.g., MIT) here.

---

## 🙋 Support

Open an issue or discussion in the repo. For environment or data-path problems, include:
- OS/Python version
- `pip freeze` or `pip list`
- Exact error messages and stack traces

---

## Running the Flask App

This project also includes a `Flask` application (`app.py`) for serving predictions via an API.

### Run Locally

```bash
# Make sure dependencies are installed
pip install -r requirements.txt  

# Start the Flask service
python app.py
```

By default, the app runs at: [http://127.0.0.1:8080](http://127.0.0.1:8080)

### Example Request

You can test the API using **curl** or **Postman**:

```bash
curl -X POST http://127.0.0.1:8080/predict     -H "Content-Type: application/json"     -d '{"text": "The product quality was excellent!"}'
```

### Example Response

```json
{
  "prediction": "Positive"
}
```

Depending on the endpoint configuration, the response can return:
- **Sentiment classification** (`Positive`, `Negative`, `Neutral`)
- **Churn prediction** (probability of churn)
- **Forecasted sales/demand values**

---


---

## 🧪 Running the Flask App (API)

If your repo includes an `app.py` (Flask) to serve models, use these steps.

### Local (Flask development server)
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Run Flask app (adjust host/port if your app differs)
python app.py
# App typically runs at http://127.0.0.1:8080 or http://127.0.0.1:5000
```

### Production-style (Gunicorn)
```bash
# Bind to 0.0.0.0:8080 so Docker/other hosts can access
gunicorn --bind 0.0.0.0:8080 app:app
```

> **Note:** In Docker, ensure your `CMD` or `ENTRYPOINT` uses a **colon** in the bind, e.g. `--bind 0.0.0.0:8080` (not a dot).

### Example Requests (adjust to your actual endpoints/payloads)

**1) Generic predict**
```bash
curl -X POST http://127.0.0.1:8080/predict   -H "Content-Type: application/json"   -d '{"text": "The product quality was excellent!"}'
```

**2) Churn scoring**
```bash
curl -X POST http://127.0.0.1:8080/churn   -H "Content-Type: application/json"   -d '{"features": {"age": 34, "gender": "Female", "region": "West", "segment": "Consumer", "tenure_months": 18}}'
```

**3) Monthly demand forecasting**
```bash
curl -X POST http://127.0.0.1:8080/forecast   -H "Content-Type: application/json"   -d '{"series_key": {"product_id": "P123", "category": "Office Supplies", "region": "West"}, "horizon": 1}'
```

**4) Sentiment**
```bash
curl -X POST http://127.0.0.1:8080/sentiment   -H "Content-Type: application/json"   -d '{"text": "Delivery was late and support didn’t help."}'
```

Return format is typically JSON. Update the endpoint names and payload keys to match your `app.py`.
