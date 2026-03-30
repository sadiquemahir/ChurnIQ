"""
Generate static PNGs for README from IBM Telco data (same pipeline style as app.py).
Run from repo root:  python scripts/generate_readme_assets.py
Requires: pip install -r requirements.txt (includes kaleido for Plotly export).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from churn_utils import clean_data, fit_label_encoders

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)

COLORS = {
    "high": "#ef4444",
    "medium": "#f59e0b",
    "low": "#10b981",
    "lr": "#3b82f6",
    "rf": "#8b5cf6",
    "xgb": "#06b6d4",
}

OUT_DIR = os.path.join(ROOT, "docs", "readme")


def _dark_layout(fig: go.Figure, *, height: int, title: str | None = None) -> None:
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(15, 20, 35, 1)",
        plot_bgcolor="rgba(10, 14, 28, 1)",
        font=dict(color="#e6ecff", size=13),
        margin=dict(t=56 if title else 40, b=48, l=56, r=32),
        height=height,
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df_raw = pd.read_csv(TELCO_URL)
    df = clean_data(df_raw)

    df_model = df.copy()
    df_model, _ = fit_label_encoders(df_model)
    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_probs = lr.predict_proba(X_test_sc)[:, 1]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    results = {
        "Logistic Regression": {"probs": lr_probs, "auc": roc_auc_score(y_test, lr_probs), "color": COLORS["lr"]},
        "Random Forest": {"probs": rf_probs, "auc": roc_auc_score(y_test, rf_probs), "color": COLORS["rf"]},
    }

    best_name = max(results, key=lambda m: results[m]["auc"])
    best_probs = results[best_name]["probs"]
    risk = pd.cut(
        best_probs,
        bins=[0, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"],
    )
    order = ["High", "Medium", "Low"]
    vals = [int((risk == lab).sum()) for lab in order]
    wedge_colors = [COLORS["high"], COLORS["medium"], COLORS["low"]]
    pie = go.Figure(
        data=[
            go.Pie(
                labels=order,
                values=vals,
                marker=dict(colors=wedge_colors, line=dict(color="rgba(230,236,255,0.2)", width=2)),
                textinfo="label+percent",
                textfont=dict(size=14, color="#f8fbff"),
                hole=0.45,
            )
        ]
    )
    _dark_layout(pie, height=420, title="Risk split (test set, best model)")
    pie.write_image(os.path.join(OUT_DIR, "risk_distribution.png"), width=900, height=480, scale=2)

    roc_fig = go.Figure()
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["probs"])
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC={res['auc']:.3f})",
                line=dict(color=res["color"], width=3),
            )
        )
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="#64748b", dash="dash", width=1.5),
        )
    )
    _dark_layout(roc_fig, height=460, title="ROC curves (IBM Telco)")
    roc_fig.update_xaxes(title="False Positive Rate", gridcolor="rgba(159,176,211,0.15)")
    roc_fig.update_yaxes(title="True Positive Rate", gridcolor="rgba(159,176,211,0.15)")
    roc_fig.update_layout(
        legend=dict(
            bgcolor="rgba(20,26,42,0.75)",
            bordercolor="rgba(159,176,211,0.25)",
            font=dict(color="#e8eeff", size=12),
        )
    )
    roc_fig.write_image(os.path.join(OUT_DIR, "roc_curves.png"), width=920, height=520, scale=2)

    eda = go.Figure()
    eda.add_trace(
        go.Histogram(
            x=df[df["Churn"] == 0]["MonthlyCharges"],
            nbinsx=16,
            marker_color="#22c55e",
            opacity=0.9,
            name="Stayed",
        )
    )
    eda.add_trace(
        go.Histogram(
            x=df[df["Churn"] == 1]["MonthlyCharges"],
            nbinsx=16,
            marker_color="#f97316",
            opacity=0.9,
            name="Churned",
        )
    )
    eda.update_layout(barmode="group", bargap=0.16)
    _dark_layout(eda, height=460, title="Monthly charges — stayed vs churned")
    eda.update_xaxes(title="Monthly Charges ($)", showgrid=False)
    eda.update_yaxes(title="Customers", gridcolor="rgba(159,176,211,0.15)")
    eda.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0.5,
            xanchor="center",
            font=dict(color="#e8eeff", size=13),
        )
    )
    eda.write_image(os.path.join(OUT_DIR, "eda_monthly_charges.png"), width=920, height=520, scale=2)

    lr_pred = lr.predict(X_test_sc)
    rf_pred = rf.predict(X_test)
    metrics_rows = [
        ["Logistic Regression", f"{accuracy_score(y_test, lr_pred)*100:.1f}%", f"{lr_probs.mean()*0.1:.4f}"],  # placeholder
    ]
    # Fixed: show real metrics
    lr_p = lr.predict(X_test_sc)
    rf_p = rf.predict(X_test)
    metrics_data = {
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [
            f"{accuracy_score(y_test, lr_p)*100:.1f}%",
            f"{accuracy_score(y_test, rf_p)*100:.1f}%",
        ],
        "AUC-ROC": [
            f"{roc_auc_score(y_test, lr_probs):.4f}",
            f"{roc_auc_score(y_test, rf_probs):.4f}",
        ],
        "F1": [
            f"{f1_score(y_test, lr_p):.4f}",
            f"{f1_score(y_test, rf_p):.4f}",
        ],
    }
    mdf = pd.DataFrame(metrics_data)
    n = len(mdf)
    row_bg = [
        "rgba(26, 34, 54, 0.96)" if i % 2 == 0 else "rgba(20, 28, 46, 0.96)"
        for i in range(n)
    ]
    fill_per_col = [[row_bg[i] for i in range(n)] for _ in range(len(mdf.columns))]
    table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[str(c) for c in mdf.columns],
                    fill_color="rgba(32, 42, 68, 0.98)",
                    font=dict(color="#e8eeff", size=14),
                    align="left",
                    height=36,
                ),
                cells=dict(
                    values=[mdf[c].tolist() for c in mdf.columns],
                    fill_color=fill_per_col,
                    font=dict(color="#d8e2f0", size=13),
                    align="left",
                    height=32,
                ),
            )
        ]
    )
    table.update_layout(
        paper_bgcolor="rgba(15, 20, 35, 1)",
        margin=dict(l=24, r=24, t=48, b=12),
        height=320,
        title=dict(text="Model comparison (hold-out test)", font=dict(color="#e6ecff", size=15)),
    )
    table.write_image(os.path.join(OUT_DIR, "metrics_table.png"), width=880, height=380, scale=2)

    print("Wrote PNGs to:", OUT_DIR)


if __name__ == "__main__":
    main()
