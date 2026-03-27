# ChurnIQ — Churn prediction dashboard

Streamlit app that trains and compares churn models (logistic regression, random forest, optional XGBoost), explores the IBM Telco dataset (or your CSV), and surfaces predictions with interactive charts.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Configuration

### Database (optional — SQL Explorer tab)

Do **not** put credentials in the code. Use either:

1. **Environment variable**

   ```bash
   set DATABASE_URL=postgresql://user:password@localhost:5432/churndb
   ```

2. **Streamlit secrets** — copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `DATABASE_URL`.

If `DATABASE_URL` is unset, the rest of the app runs; only the SQL Explorer will error until you configure it.

## Docker

```bash
docker build -t churniq .
docker run -p 8501:8501 -e DATABASE_URL=postgresql://... churniq
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -q
```

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI, training, evaluation |
| `churn_utils.py` | `clean_data`, persisted `LabelEncoder` helpers |
| `tests/` | Pytest coverage for data prep |

## Features

- **Consistent encoding**: single-customer predictions use the same encoders as training (no refit-on-the-fly leakage).
- **Thresholds & precision/recall**: adjustable classification threshold with simple cost weights for false negatives vs false positives.
- **Calibration plot** for the best model.
- **Permutation importance** and **SHAP** (Random Forest) for explainability.
