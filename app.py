import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              confusion_matrix, roc_curve, classification_report)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ — Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #070b12;
    color: #c8d6e5;
}
.stApp { background-color: #070b12; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1421 !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d6e5 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0d1421;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { color: #00ff94 !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #4a6fa5 !important; font-size: 0.7rem !important; letter-spacing: 2px; text-transform: uppercase; }
[data-testid="stMetricDelta"] { color: #ffe66d !important; }

/* Buttons */
.stButton > button {
    background: rgba(0,255,148,0.08) !important;
    border: 1px solid rgba(0,255,148,0.4) !important;
    color: #00ff94 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,255,148,0.18) !important;
    border-color: #00ff94 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1421;
    border-bottom: 1px solid #1e3a5f;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a6fa5 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #00ff94 !important;
    border-bottom: 2px solid #00ff94 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; }

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
h1 { color: #ffffff !important; }
h2 { color: #00ff94 !important; font-size: 1.1rem !important; letter-spacing: 2px; text-transform: uppercase; }
h3 { color: #c8d6e5 !important; font-size: 0.95rem !important; }

/* Selectbox / inputs */
[data-baseweb="select"] { background: #0d1421 !important; border-color: #1e3a5f !important; }
.stSelectbox label, .stSlider label, .stFileUploader label { color: #4a6fa5 !important; font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }

/* Progress bar */
.stProgress > div > div { background: #00ff94 !important; }

/* Divider */
hr { border-color: #1e3a5f !important; }

/* Alert boxes */
.stAlert { background: #0d1421 !important; border: 1px solid #1e3a5f !important; border-radius: 8px; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1421;
    border: 1px dashed #1e3a5f;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Plot theme ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1421", "axes.facecolor": "#0d1421",
    "axes.edgecolor": "#1e3a5f", "text.color": "#c8d6e5",
    "axes.labelcolor": "#c8d6e5", "xtick.color": "#4a6fa5",
    "ytick.color": "#4a6fa5", "grid.color": "#1e3a5f",
    "font.family": "monospace", "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {"high": "#ff6b6b", "medium": "#ffe66d", "low": "#00ff94",
          "lr": "#4ecdc4", "rf": "#ffe66d", "xgb": "#a78bfa", "accent": "#00ff94"}


# ── Helpers ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_default_data():
    URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    try:
        return pd.read_csv(URL)
    except Exception:
        return None


def clean_data(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["Churn"].isnull().any():
        df["Churn"] = df["Churn"].fillna(0).astype(int)
    if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype in [int, float]:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
    return df


@st.cache_resource
def train_models(df):
    df_model = df.copy()
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_probs = lr.predict_proba(X_test_sc)[:, 1]
    lr_preds = lr.predict(X_test_sc)
    results["Logistic Regression"] = {
        "model": lr, "probs": lr_probs, "preds": lr_preds,
        "accuracy": accuracy_score(y_test, lr_preds),
        "auc": roc_auc_score(y_test, lr_probs),
        "f1": f1_score(y_test, lr_preds), "color": COLORS["lr"],
        "scaled": True,
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = rf.predict(X_test)
    results["Random Forest"] = {
        "model": rf, "probs": rf_probs, "preds": rf_preds,
        "accuracy": accuracy_score(y_test, rf_preds),
        "auc": roc_auc_score(y_test, rf_probs),
        "f1": f1_score(y_test, rf_preds), "color": COLORS["rf"],
        "scaled": False,
    }

    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                             eval_metric="logloss", random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]
        xgb_preds = xgb.predict(X_test)
        results["XGBoost"] = {
            "model": xgb, "probs": xgb_probs, "preds": xgb_preds,
            "accuracy": accuracy_score(y_test, xgb_preds),
            "auc": roc_auc_score(y_test, xgb_probs),
            "f1": f1_score(y_test, xgb_preds), "color": COLORS["xgb"],
            "scaled": False,
        }
    except ImportError:
        pass

    # Build pred_df using best model
    best_name = max(results, key=lambda m: results[m]["auc"])
    best = results[best_name]
    pred_df = X_test.copy().reset_index(drop=True)
    pred_df.index.name = "Customer_ID"
    pred_df["Actual_Churn"]      = y_test.values
    pred_df["Churn_Probability"] = best["probs"]
    pred_df["Predicted_Churn"]   = best["preds"]
    pred_df["Risk_Level"] = pd.cut(
        pred_df["Churn_Probability"],
        bins=[0, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"]
    )

    feature_importance = pd.Series(
        rf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return results, pred_df, feature_importance, best_name, X_test, y_test, scaler, X.columns.tolist()


def risk_color(score):
    if score >= 0.65: return COLORS["high"]
    if score >= 0.35: return COLORS["medium"]
    return COLORS["low"]


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 28px 0 8px 0; display:flex; align-items:center; gap:14px;">
  <div style="width:12px;height:12px;border-radius:50%;background:#00ff94;box-shadow:0 0 14px #00ff94;animation:pulse 2s infinite;"></div>
  <span style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#fff;letter-spacing:-1px;">
    Churn<span style="color:#00ff94;">IQ</span>
  </span>
  <span style="font-size:11px;color:#4a6fa5;border-left:1px solid #1e3a5f;padding-left:14px;letter-spacing:1px;">
    SaaS CHURN PREDICTION · IBM TELCO DATASET
  </span>
</div>
<hr style="margin-bottom:24px;">
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Dataset")
    data_source = st.radio("", ["Use IBM Telco Dataset (auto)", "Upload my own CSV"], label_visibility="collapsed")
    df_raw = None

    if data_source == "Use IBM Telco Dataset (auto)":
        with st.spinner("Loading dataset..."):
            df_raw = load_default_data()
        if df_raw is not None:
            st.success(f"✅ {len(df_raw):,} rows loaded")
        else:
            st.error("❌ Could not load. Check internet connection.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ {len(df_raw):,} rows loaded")
        else:
            st.info("Upload a CSV with a 'Churn' column (Yes/No or 1/0)")

    st.markdown("---")
    st.markdown("## 🤖 Models")
    st.markdown("""
<div style="font-size:11px;color:#4a6fa5;line-height:2;">
🟦 Logistic Regression<br>
🟨 Random Forest<br>
🟪 XGBoost (if installed)<br>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style="font-size:10px;color:#2a4a6f;line-height:1.8;">
Built for portfolio showcase.<br>
IBM Telco · scikit-learn · Streamlit<br><br>
<span style="color:#00ff94;">github.com/yourhandle</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — only show when data is loaded
# ══════════════════════════════════════════════════════════════════════════════
if df_raw is None:
    st.markdown("""
    <div style="text-align:center;padding:80px 0;color:#2a4a6f;">
        <div style="font-size:48px;margin-bottom:16px;">🔮</div>
        <div style="font-size:14px;letter-spacing:2px;">LOAD A DATASET TO BEGIN</div>
        <div style="font-size:11px;margin-top:8px;">Use the sidebar to auto-load the IBM Telco dataset or upload your own CSV</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Clean data
df = clean_data(df_raw)

# Train
with st.spinner("🤖 Training models... this takes ~15 seconds"):
    results, pred_df, feat_imp, best_name, X_test, y_test, scaler, feature_cols = train_models(df)

best = results[best_name]

# ── TABS ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 EDA", "🤖 Model Comparison", "🎯 Predictions", "🔮 Predict a Customer"
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    churn_rate = df["Churn"].mean() * 100
    high_risk  = (pred_df["Risk_Level"] == "High").sum()
    retained_mrr = df_raw[df["Churn"] == 0]["MonthlyCharges"].sum() if "MonthlyCharges" in df_raw.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL CUSTOMERS",  f"{len(df):,}")
    c2.metric("CHURN RATE",       f"{churn_rate:.1f}%",    delta=f"{df['Churn'].sum()} churned")
    c3.metric("HIGH RISK (test)", f"{high_risk}",          delta="need attention", delta_color="inverse")
    c4.metric("BEST MODEL AUC",   f"{best['auc']:.3f}",   delta=best_name)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("## Risk Distribution")
        risk_counts = pred_df["Risk_Level"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        wedge_colors = [{"High": COLORS["high"], "Medium": COLORS["medium"], "Low": COLORS["low"]}[l]
                        for l in risk_counts.index]
        ax.pie(risk_counts.values, labels=risk_counts.index, colors=wedge_colors,
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "#0d1421", "linewidth": 3},
               textprops={"color": "#c8d6e5", "fontsize": 11})
        ax.set_facecolor("#0d1421")
        fig.patch.set_facecolor("#0d1421")
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("## Model Leaderboard")
        for name, res in sorted(results.items(), key=lambda x: -x[1]["auc"]):
            crown = "🏆 " if name == best_name else "   "
            pct = int(res["auc"] * 100)
            st.markdown(f"""
            <div style="margin-bottom:14px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="font-size:12px;color:#c8d6e5;">{crown}{name}</span>
                <span style="font-size:12px;color:{res['color']};">AUC {res['auc']:.3f}</span>
              </div>
              <div style="height:6px;background:#1e3a5f;border-radius:3px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{res['color']};border-radius:3px;box-shadow:0 0 8px {res['color']};"></div>
              </div>
              <div style="display:flex;gap:20px;margin-top:5px;">
                <span style="font-size:10px;color:#4a6fa5;">Acc {res['accuracy']*100:.1f}%</span>
                <span style="font-size:10px;color:#4a6fa5;">F1 {res['f1']:.3f}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Exploratory Data Analysis")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Churn Pattern Analysis", fontsize=13, color="#00ff94", y=1.01)

    # 1. Contract Type
    if "Contract" in df.columns:
        ct = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
        bars = axes[0,0].bar(ct.index, ct.values, color=[COLORS["high"], COLORS["medium"], COLORS["low"]], edgecolor="none", width=0.5)
        axes[0,0].set_title("Churn Rate by Contract", color="#c8d6e5")
        axes[0,0].set_ylabel("Churn Rate (%)")
        axes[0,0].set_ylim(0, max(ct.values) * 1.3)
        axes[0,0].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, ct.values):
            axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", color="#c8d6e5", fontsize=9)

    # 2. Monthly Charges
    if "MonthlyCharges" in df.columns:
        axes[0,1].hist(df[df["Churn"]==0]["MonthlyCharges"], bins=30, alpha=0.75, color=COLORS["low"], label="Retained", edgecolor="none")
        axes[0,1].hist(df[df["Churn"]==1]["MonthlyCharges"], bins=30, alpha=0.75, color=COLORS["high"], label="Churned", edgecolor="none")
        axes[0,1].set_title("Monthly Charges", color="#c8d6e5")
        axes[0,1].set_xlabel("Monthly Charges ($)")
        axes[0,1].legend(facecolor="#1e3a5f", edgecolor="none", labelcolor="#c8d6e5")
        axes[0,1].grid(axis="y", alpha=0.3)

    # 3. Internet Service
    if "InternetService" in df.columns:
        it = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False) * 100
        bars2 = axes[0,2].bar(it.index, it.values, color=[COLORS["high"], COLORS["medium"], COLORS["low"]], edgecolor="none", width=0.5)
        axes[0,2].set_title("Churn by Internet Service", color="#c8d6e5")
        axes[0,2].set_ylim(0, max(it.values) * 1.3)
        axes[0,2].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars2, it.values):
            axes[0,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", color="#c8d6e5", fontsize=9)

    # 4. Tenure
    if "tenure" in df.columns:
        axes[1,0].hist(df[df["Churn"]==0]["tenure"], bins=30, alpha=0.75, color=COLORS["low"], label="Retained", edgecolor="none")
        axes[1,0].hist(df[df["Churn"]==1]["tenure"], bins=30, alpha=0.75, color=COLORS["high"], label="Churned", edgecolor="none")
        axes[1,0].set_title("Tenure Distribution", color="#c8d6e5")
        axes[1,0].set_xlabel("Tenure (months)")
        axes[1,0].legend(facecolor="#1e3a5f", edgecolor="none", labelcolor="#c8d6e5")
        axes[1,0].grid(axis="y", alpha=0.3)

    # 5. Churn counts
    churn_counts = df["Churn"].value_counts()
    axes[1,1].bar(["Retained", "Churned"], churn_counts.values,
                  color=[COLORS["low"], COLORS["high"]], edgecolor="none", width=0.5)
    axes[1,1].set_title("Churn Counts", color="#c8d6e5")
    axes[1,1].grid(axis="y", alpha=0.3)
    for i, val in enumerate(churn_counts.values):
        axes[1,1].text(i, val+20, f"{val:,}", ha="center", color="#c8d6e5", fontsize=10)

    # 6. Correlation
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=axes[1,2], cmap="RdYlGn", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.5, linecolor="#0d1421", cbar=False)
    axes[1,2].set_title("Correlation Matrix", color="#c8d6e5")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Key stats
    st.markdown("---")
    st.markdown("## 📌 Key Insights")
    cols = st.columns(3)
    if "Contract" in df.columns:
        mtm = df[df["Contract"]=="Month-to-month"]["Churn"].mean()*100
        twyr = df[df["Contract"]=="Two year"]["Churn"].mean()*100
        cols[0].markdown(f"""<div style="background:#0d1421;border:1px solid #1e3a5f;border-radius:8px;padding:16px;">
        <div style="font-size:9px;color:#4a6fa5;letter-spacing:2px;">CONTRACT IMPACT</div>
        <div style="font-size:20px;color:#ff6b6b;font-weight:700;margin:8px 0;">{mtm:.0f}% vs {twyr:.0f}%</div>
        <div style="font-size:11px;color:#8aadcc;">Month-to-month vs 2-year churn</div></div>""", unsafe_allow_html=True)
    if "tenure" in df.columns:
        avg_churned = df[df["Churn"]==1]["tenure"].mean()
        avg_retained = df[df["Churn"]==0]["tenure"].mean()
        cols[1].markdown(f"""<div style="background:#0d1421;border:1px solid #1e3a5f;border-radius:8px;padding:16px;">
        <div style="font-size:9px;color:#4a6fa5;letter-spacing:2px;">AVG TENURE</div>
        <div style="font-size:20px;color:#ffe66d;font-weight:700;margin:8px 0;">{avg_churned:.0f}mo vs {avg_retained:.0f}mo</div>
        <div style="font-size:11px;color:#8aadcc;">Churned vs retained customers</div></div>""", unsafe_allow_html=True)
    cols[2].markdown(f"""<div style="background:#0d1421;border:1px solid #1e3a5f;border-radius:8px;padding:16px;">
    <div style="font-size:9px;color:#4a6fa5;letter-spacing:2px;">DATASET SIZE</div>
    <div style="font-size:20px;color:#00ff94;font-weight:700;margin:8px 0;">{len(df):,}</div>
    <div style="font-size:11px;color:#8aadcc;">{df.shape[1]} features · {df['Churn'].sum()} churned</div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Model Performance Comparison")

    # Metrics table
    metrics_data = {
        "Model": list(results.keys()),
        "Accuracy": [f"{r['accuracy']*100:.1f}%" for r in results.values()],
        "AUC-ROC":  [f"{r['auc']:.4f}" for r in results.values()],
        "F1 Score": [f"{r['f1']:.4f}" for r in results.values()],
        "Best":     ["🏆" if n == best_name else "" for n in results.keys()],
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## ROC Curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res["probs"])
            lw = 3 if name == best_name else 1.5
            ax.plot(fpr, tpr, color=res["color"], lw=lw, label=f"{name}  AUC={res['auc']:.3f}")
        ax.plot([0,1],[0,1], "--", color="#4a6fa5", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(facecolor="#1e3a5f", edgecolor="none", labelcolor="#c8d6e5", fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("## Feature Importance (Random Forest)")
        top12 = feat_imp.head(12).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        colors_fi = [COLORS["high"] if v > top12.median() else COLORS["lr"] for v in top12.values]
        ax.barh(top12.index, top12.values, color=colors_fi, edgecolor="none", height=0.6)
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        for i, val in enumerate(top12.values):
            ax.text(val+0.001, i, f"{val:.3f}", va="center", fontsize=8, color="#c8d6e5")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("## Confusion Matrix — Best Model")
    cm = confusion_matrix(y_test, best["preds"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="YlOrRd", cbar=False,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"],
                linewidths=2, linecolor="#0d1421",
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}", color="#c8d6e5")
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"## Predictions — {best_name} (Best Model)")

    col1, col2, col3 = st.columns(3)
    col1.metric("HIGH RISK",   f"{(pred_df['Risk_Level']=='High').sum()}")
    col2.metric("MEDIUM RISK", f"{(pred_df['Risk_Level']=='Medium').sum()}")
    col3.metric("LOW RISK",    f"{(pred_df['Risk_Level']=='Low').sum()}")

    st.markdown("---")
    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        st.markdown("## Top 20 At-Risk Customers")
        top20 = pred_df.nlargest(20, "Churn_Probability")
        fig, ax = plt.subplots(figsize=(7, 7))
        bar_colors = [COLORS["high"] if r=="High" else COLORS["medium"] for r in top20["Risk_Level"]]
        ax.barh(range(len(top20)), top20["Churn_Probability"]*100,
                color=bar_colors, edgecolor="none", height=0.7)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([f"Customer {i}" for i in top20.index], fontsize=8)
        ax.set_xlabel("Churn Probability (%)")
        ax.axvline(65, color=COLORS["medium"], linestyle="--", lw=1, alpha=0.7)
        ax.set_xlim(0, 108)
        ax.grid(axis="x", alpha=0.3)
        for i, (idx, row) in enumerate(top20.iterrows()):
            ax.text(row["Churn_Probability"]*100+0.5, i, f"{row['Churn_Probability']*100:.1f}%",
                    va="center", fontsize=7, color="#c8d6e5")
        handles = [mpatches.Patch(color=COLORS["high"], label="High"),
                   mpatches.Patch(color=COLORS["medium"], label="Medium")]
        ax.legend(handles=handles, facecolor="#1e3a5f", edgecolor="none",
                  labelcolor="#c8d6e5", loc="lower right", fontsize=9)
        st.pyplot(fig)
        plt.close()

    with col_table:
        st.markdown("## All Predictions")
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
        display_df = pred_df[["Churn_Probability", "Risk_Level", "Predicted_Churn", "Actual_Churn"]].copy()
        display_df["Churn_Probability"] = (display_df["Churn_Probability"] * 100).round(1).astype(str) + "%"
        if risk_filter != "All":
            display_df = display_df[display_df["Risk_Level"] == risk_filter]
        st.dataframe(display_df, use_container_width=True, height=420)

        csv = pred_df[["Churn_Probability", "Risk_Level", "Predicted_Churn", "Actual_Churn"]].to_csv()
        st.download_button("⬇ Download Predictions CSV", csv, "churn_predictions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — PREDICT A SINGLE CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔮 Predict a Single Customer")
    st.markdown("<div style='font-size:12px;color:#4a6fa5;margin-bottom:24px;'>Adjust the sliders to simulate a customer profile and get an instant churn prediction.</div>", unsafe_allow_html=True)

    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        tenure         = st.slider("Tenure (months)", 0, 72, 12)
        monthly        = st.slider("Monthly Charges ($)", 18, 120, 65)
        total          = st.slider("Total Charges ($)", 0, 9000, monthly * tenure)
        contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet       = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment        = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner        = st.selectbox("Has Partner", ["Yes", "No"])
        dependents     = st.selectbox("Has Dependents", ["No", "Yes"])
        phone          = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        online_sec     = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support   = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv   = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col_result:
        if st.button("RUN PREDICTION →"):
            # Build a row matching the training features
            sample = pd.DataFrame([{
                "gender": "Male", "SeniorCitizen": senior,
                "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone,
                "MultipleLines": multiple_lines, "InternetService": internet,
                "OnlineSecurity": online_sec, "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": tech_support,
                "StreamingTV": streaming_tv, "StreamingMovies": "No",
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly, "TotalCharges": total,
            }])

            # Encode same as training
            df_enc = df.drop("Churn", axis=1).copy()
            sample_full = pd.concat([df_enc, sample], ignore_index=True)
            for col in sample_full.select_dtypes(include="object").columns:
                sample_full[col] = LabelEncoder().fit_transform(sample_full[col].astype(str))
            sample_encoded = sample_full.iloc[[-1]][feature_cols]

            rf_model = results["Random Forest"]["model"]
            prob = rf_model.predict_proba(sample_encoded)[0][1]
            risk = "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low"
            rcolor = COLORS["high"] if risk=="High" else COLORS["medium"] if risk=="Medium" else COLORS["low"]

            st.markdown(f"""
            <div style="background:#0d1421;border:1px solid {rcolor};border-radius:12px;padding:32px;text-align:center;margin-top:16px;">
                <div style="font-size:11px;color:#4a6fa5;letter-spacing:2px;margin-bottom:16px;">CHURN PROBABILITY</div>
                <div style="font-size:64px;font-weight:800;color:{rcolor};font-family:'Syne',sans-serif;line-height:1;">
                    {prob*100:.1f}%
                </div>
                <div style="margin:16px 0;padding:8px 24px;background:rgba(0,0,0,0.3);border-radius:20px;display:inline-block;">
                    <span style="font-size:13px;font-weight:700;color:{rcolor};letter-spacing:3px;">{risk.upper()} RISK</span>
                </div>
                <div style="margin-top:24px;font-size:11px;color:#4a6fa5;line-height:2;text-align:left;">
                    {'⚠️ High ticket to churn. Consider proactive outreach.' if risk=='High' else
                     '👀 Monitor this account. Check engagement metrics.' if risk=='Medium' else
                     '✅ This customer looks healthy. Keep up the good work.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk signals
            st.markdown("#### Risk Signals")
            signals = [
                ("Short tenure", tenure < 12, f"{tenure} months — high churn window"),
                ("Month-to-month contract", contract == "Month-to-month", "Highest churn segment"),
                ("Fiber optic (high cost)", internet == "Fiber optic", "Higher churn despite higher spend"),
                ("Electronic check payment", payment == "Electronic check", "Correlated with churn"),
                ("No tech support", tech_support == "No", "Unresolved issues drive churn"),
                ("High monthly charges", monthly > 80, f"${monthly}/mo — above avg"),
            ]
            for label, triggered, note in signals:
                icon = "🔴" if triggered else "🟢"
                color = "#ff6b6b" if triggered else "#4a6fa5"
                st.markdown(f"""<div style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #1a2840;font-size:12px;">
                <span>{icon}</span>
                <span style="color:{color};flex:1;">{label}</span>
                {f'<span style="color:#4a6fa5;font-size:10px;">{note}</span>' if triggered else ''}
                </div>""", unsafe_allow_html=True)
                