# ChurnIQ — Churn prediction dashboard

[![CI](https://github.com/sadiquemahir/churn-project/actions/workflows/ci.yml/badge.svg)](https://github.com/sadiquemahir/churn-project/actions/workflows/ci.yml)

**ChurnIQ** is a decision-support web app for **retention and analytics teams**: it estimates churn risk from tabular customer data, compares models, and explains *which factors* drive scores so teams can prioritize outreach—not just stare at accuracy tables.

Public demo (when deployed): use **Streamlit Community Cloud** from this repo (`app.py` as entrypoint).

---

## Problem → solution

| Business need | What ChurnIQ does |
|---------------|-------------------|
| Prioritize who to contact first | Risk tiers (High / Medium / Low) on held-out test predictions |
| Balance false alarms vs missed churners | Adjustable probability **threshold** and simple **FN vs FP cost** framing |
| Trust the score | **Calibration** curve, **permutation** and **SHAP**-style explanations (Random Forest) |
| Use your own data | Sidebar: IBM Telco sample or **upload CSV** with a `Churn` column |
| Optional warehouse analytics | **SQL Explorer** tab when `DATABASE_URL` points to Postgres (read-only role recommended in production) |

---

## Architecture

High-level data flow (training and scoring run in the Streamlit process; no separate API in this repo—suitable for demos; production would often split serving from training).

```mermaid
flowchart LR
  subgraph inputs [Inputs]
    CSV[CSV upload / Telco sample]
    DB[(PostgreSQL optional)]
  end

  subgraph app [Streamlit app]
    CLEAN[clean_data + encoders]
    TRAIN[Train / compare models]
    UI[Charts, tables, thresholds]
  end

  CSV --> CLEAN
  DB --> UI
  CLEAN --> TRAIN
  TRAIN --> UI
```

---

## Reproducibility

- **Python**: **3.11** (see `.python-version`; matches `Dockerfile` base image).
- **Install**: always use a **virtual environment** and `pip install -r requirements.txt` so your laptop matches CI and Docker.
- **Run**: `streamlit run app.py` — training runs on load for the loaded dataset (demo-style; production would typically pre-train and load artifacts).

---

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Configuration

### Database (optional — SQL Explorer tab)

Do **not** put credentials in the code. Use either:

1. **Environment variable**

   ```bash
   set DATABASE_URL=postgresql://user:password@localhost:5432/churndb
   ```

2. **Streamlit secrets** — copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `DATABASE_URL`.

If `DATABASE_URL` is unset, the rest of the app runs; only the SQL Explorer will error until you configure it.

---

## Docker

```bash
docker build -t churniq .
docker run -p 8501:8501 -e DATABASE_URL=postgresql://... churniq
```

---

## Tests & CI

**GitHub Actions** runs on every push/PR to `main`: install dependencies, `compileall` on `app.py` / `churn_utils.py`, and `pytest`.

```bash
pip install -r requirements.txt
pytest tests/ -q
```

---

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI, training, evaluation, charts |
| `churn_utils.py` | `clean_data`, persisted `LabelEncoder` helpers |
| `tests/` | Pytest coverage for data prep and encoding |
| `.github/workflows/ci.yml` | CI pipeline |

---

## Features (technical)

- **Consistent encoding**: single-customer predictions use the same encoders as training (no refit-on-the-fly leakage).
- **Thresholds & precision/recall**: adjustable classification threshold with simple cost weights for false negatives vs false positives.
- **Calibration plot** for the best model.
- **Permutation importance** and **SHAP** (Random Forest, requires `shap` installed) for explainability.

---

## Limitations & production considerations

- **Dataset**: IBM Telco is a classic benchmark; real churn populations drift—models should be **retrained** and **monitored** on a schedule.
- **Leakage**: Features must be known **at prediction time** (no future invoices in features). This demo uses standard columns; custom CSVs need the same discipline.
- **Serving**: This app retrains in-process for simplicity. In production you would typically **version models**, serve via an API or batch jobs, and **log** inputs/outputs for audit and drift detection.
- **Security**: Use **secrets managers**, **read-only** DB users for analytics, and **no PII** in public screenshots or sample data.

### What you would add in a real deployment

- **Monitoring**: prediction volume, latency, label delay, **calibration drift**, feature distribution drift.
- **Governance**: model cards, approval workflow, access control (SSO), and retention policies for logs.
- **Testing**: broader integration tests and load tests on the serving path (beyond this repo’s data-prep unit tests).

---

## Deploy as a website

Connect this GitHub repo to **[Streamlit Community Cloud](https://share.streamlit.io)** (entry file `app.py`, branch `main`). You get a public HTTPS URL for stakeholders to try the app.

---

## License

Add a `LICENSE` file if you open-source or share the repo publicly.
