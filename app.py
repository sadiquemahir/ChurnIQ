import os
from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              confusion_matrix, roc_curve, precision_score,
                              recall_score)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from sqlalchemy import create_engine, text

from churn_utils import clean_data, fit_label_encoders, encode_row_with_encoders


def _database_url() -> Optional[str]:
    u = os.environ.get("DATABASE_URL")
    if u:
        return u
    try:
        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"])
    except Exception:
        pass
    return None


_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        url = _database_url()
        if url:
            _engine = create_engine(url)
    return _engine


def load_from_db(query):
    eng = _get_engine()
    if eng is None:
        raise RuntimeError(
            "Database not configured. Set DATABASE_URL in environment or .streamlit/secrets.toml"
        )
    with eng.connect() as conn:
        return pd.read_sql(text(query), conn)

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

:root{
  --bg0:#080b12;
  --bg1:#0b1020;
  --panel: rgba(20, 26, 42, 0.72);
  --panel2: rgba(20, 26, 42, 0.88);
  --stroke: rgba(99, 132, 255, 0.24);
  --stroke2: rgba(147, 120, 255, 0.20);
  --text:#e6ecff;
  --muted:#9fb0d3;
  --accent:#2563eb;
  --warn:#f59e0b;
  --danger:#ef4444;
  --shadow: 0 24px 56px rgba(2, 6, 23, 0.60);
  --shadow2: 0 12px 32px rgba(2, 6, 23, 0.50);
  --radius: 18px;
}

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; background: var(--bg1); color: var(--text); }
.stApp {
  color-scheme: dark;
  background: radial-gradient(1100px 700px at 8% -10%, rgba(59,130,246,0.24), transparent 55%),
                 radial-gradient(920px 620px at 95% 8%, rgba(168,85,247,0.20), transparent 58%),
                 radial-gradient(980px 720px at 50% 120%, rgba(14,165,233,0.16), transparent 60%),
                 linear-gradient(180deg, var(--bg0), var(--bg1));
}

/* 2026 background: animated aurora + subtle noise */
.stApp::before{
  content:"";
  position: fixed;
  inset: -20vh -10vw;
  pointer-events:none;
  z-index: 0;
  background:
    radial-gradient(600px 420px at 25% 20%, rgba(59,130,246,0.20), transparent 55%),
    radial-gradient(540px 380px at 75% 30%, rgba(168,85,247,0.20), transparent 58%),
    radial-gradient(680px 460px at 55% 80%, rgba(14,165,233,0.15), transparent 60%);
  filter: blur(14px) saturate(120%);
  opacity: 0.95;
  animation: aurora 14s ease-in-out infinite alternate;
}
.stApp::after{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events:none;
  z-index: 0;
  background-image:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='240' height='240' filter='url(%23n)' opacity='.18'/%3E%3C/svg%3E");
  mix-blend-mode: overlay;
  opacity: 0.18;
}

/* Ensure app content floats above background layers */
[data-testid="stAppViewContainer"]{ position: relative; z-index: 1; }

@keyframes aurora{
  0%{ transform: translate3d(-1.5%, -1.0%, 0) scale(1.02); }
  50%{ transform: translate3d(1.5%, 0.8%, 0) scale(1.05); }
  100%{ transform: translate3d(-0.8%, 1.2%, 0) scale(1.03); }
}
@keyframes pulse{
  0%,100%{ transform: scale(1); opacity: 0.95; }
  50%{ transform: scale(1.35); opacity: 0.65; }
}

/* Smooth typography rendering */
* { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(16,22,38,0.86), rgba(16,22,38,0.68)) !important;
    backdrop-filter: blur(14px);
    border-right: 1px solid var(--stroke);
}
[data-testid="stSidebar"] * { color: #dbe8ff !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(180deg, var(--panel2), var(--panel));
    border: 1px solid var(--stroke);
    border-radius: var(--radius);
    padding: 16px !important;
    box-shadow: var(--shadow2);
    transform: translateZ(0);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before{
    content:"";
    position:absolute;
    inset: -1px;
    background: radial-gradient(240px 120px at 20% 15%, rgba(0,255,148,0.20), transparent 55%),
                radial-gradient(260px 150px at 85% 30%, rgba(167,139,250,0.16), transparent 58%);
    opacity: 0.55;
    pointer-events:none;
}
[data-testid="metric-container"]:hover{
    transform: perspective(900px) rotateX(4deg) rotateY(-6deg) translateY(-3px);
    box-shadow: var(--shadow);
    border-color: rgba(0,255,148,0.35);
}
[data-testid="stMetricValue"] { color: #1d4ed8 !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #8fa6cf !important; font-size: 0.7rem !important; letter-spacing: 2px; text-transform: uppercase; }
[data-testid="stMetricDelta"] { color: #d97706 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(180deg, rgba(37,99,235,0.16), rgba(59,130,246,0.08)) !important;
    border: 1px solid rgba(37,99,235,0.35) !important;
    color: var(--accent) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    border-radius: 999px !important;
    padding: 10px 24px !important;
    box-shadow: 0 10px 24px rgba(37,99,235,0.16) !important;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(180deg, rgba(37,99,235,0.26), rgba(59,130,246,0.12)) !important;
    border-color: rgba(37,99,235,0.62) !important;
    box-shadow: 0 18px 44px rgba(37,99,235,0.24) !important;
}

/* Download buttons — match dark UI (default Streamlit styling can be white with unreadable text) */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(180deg, rgba(37,99,235,0.22), rgba(59,130,246,0.10)) !important;
    border: 1px solid rgba(37,99,235,0.45) !important;
    color: #e8eeff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
    border-radius: 999px !important;
    padding: 10px 22px !important;
    box-shadow: 0 10px 24px rgba(37,99,235,0.14) !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(180deg, rgba(37,99,235,0.32), rgba(59,130,246,0.14)) !important;
    border-color: rgba(37,99,235,0.65) !important;
    color: #ffffff !important;
}

/* Plotly SVG legends — keep label fill light (global CSS can darken SVG text) */
.js-plotly-plot .legendtext { fill: #e8eeff !important; }
.js-plotly-plot .legend > g > text { fill: #e8eeff !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(180deg, rgba(13,20,33,0.72), rgba(13,20,33,0.44));
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--stroke);
    gap: 4px;
    border-radius: 14px;
    padding: 6px 8px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a6fa5 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 12px !important;
    transition: background .18s ease, transform .18s ease, color .18s ease;
}
.stTabs [data-baseweb="tab"]:hover{
    background: rgba(56,189,248,0.08) !important;
    transform: translateY(-1px);
}
.stTabs [aria-selected="true"] {
    color: #1d4ed8 !important;
    background: rgba(37,99,235,0.10) !important;
    box-shadow: inset 0 0 0 1px rgba(37,99,235,0.22);
}

/* ── Dataframe & tables (dark glass, not white) ───────────────────────────── */
[data-testid="stDataFrame"] {
  --gdg-bg-cell: rgba(22, 30, 48, 0.98);
  --gdg-bg-header: rgba(28, 38, 58, 0.98);
  --gdg-bg-header-has-focus: rgba(36, 48, 72, 0.98);
  --gdg-bg-header-selected: rgba(37, 99, 235, 0.35);
  --gdg-border-color: rgba(99, 132, 255, 0.22);
  --gdg-text-dark: #e8eeff;
  --gdg-text-medium: #b8c5e8;
  --gdg-text-light: #8fa6cf;
  border: 1px solid rgba(99,132,255,0.28) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow2) !important;
  background: rgba(18, 24, 40, 0.92) !important;
}
[data-testid="stDataFrame"] > div {
  background: rgba(18, 24, 40, 0.88) !important;
}
/* Glide Data Grid (Streamlit dataframe engine) */
[data-testid="stDataFrame"] [class*="glide-data-grid"],
[data-testid="stDataFrame"] [class*="dvn-container"] {
  --gdg-bg-cell: rgba(22, 30, 48, 0.98) !important;
  --gdg-bg-header: rgba(28, 38, 58, 0.98) !important;
  --gdg-bg-header-has-focus: rgba(36, 48, 72, 0.98) !important;
  --gdg-bg-header-selected: rgba(37, 99, 235, 0.35) !important;
  --gdg-border-color: rgba(99, 132, 255, 0.22) !important;
  --gdg-text-dark: #e8eeff !important;
  --gdg-text-medium: #b8c5e8 !important;
  --gdg-text-light: #8fa6cf !important;
  background: rgba(18, 24, 40, 0.95) !important;
}

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
h1 { color: #f8fbff !important; }
h2 { color: #1d4ed8 !important; font-size: 1.1rem !important; letter-spacing: 2px; text-transform: uppercase; }
h3 { color: #d5e3ff !important; font-size: 0.95rem !important; }

/* Selectbox / dropdowns (Base Web) */
[data-baseweb="select"] {
  background: rgba(22, 30, 48, 0.92) !important;
  border-color: rgba(99, 132, 255, 0.35) !important;
  backdrop-filter: blur(10px);
  border-radius: 12px !important;
}
[data-baseweb="select"] > div {
  background-color: rgba(22, 30, 48, 0.95) !important;
  border-color: rgba(99, 132, 255, 0.35) !important;
  color: #e8eeff !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div[aria-selected] {
  color: #e8eeff !important;
}
/* Dropdown menu popover (Base Web renders in a portal — must override white panels everywhere) */
[data-baseweb="layer"],
[data-baseweb="layer"] > div {
  z-index: 100002 !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div {
  background-color: rgba(22, 30, 48, 0.98) !important;
  color: #e8eeff !important;
  border-color: rgba(99, 132, 255, 0.4) !important;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45) !important;
}
[data-baseweb="menu"],
[data-baseweb="menu"] ul,
[data-baseweb="menu"] li,
ul[role="listbox"],
[role="listbox"] {
  background-color: rgba(22, 30, 48, 0.98) !important;
  color: #e8eeff !important;
  border-radius: 12px !important;
}
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] ul {
  background-color: rgba(22, 30, 48, 0.98) !important;
  border: 1px solid rgba(99, 132, 255, 0.35) !important;
}
/* Options: force readable text on dark bg (fixes pale-on-white) */
[data-baseweb="popover"] li,
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
li[role="option"] {
  color: #e8eeff !important;
  background-color: rgba(22, 30, 48, 0.98) !important;
  border-color: transparent !important;
}
[data-baseweb="popover"] li span,
[data-baseweb="popover"] [role="option"] span,
[data-baseweb="menu"] li span {
  color: #e8eeff !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li:hover,
li[role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"] {
  background-color: rgba(37, 99, 235, 0.42) !important;
  color: #ffffff !important;
}
[data-baseweb="popover"] [aria-selected="true"] span {
  color: #ffffff !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] input:focus {
  background: rgba(22, 30, 48, 0.95) !important;
  color: #e8eeff !important;
  border: 1px solid rgba(99, 132, 255, 0.35) !important;
  border-radius: 10px !important;
  caret-color: #93c5fd !important;
}
[data-testid="stNumberInput"] button {
  background: rgba(30, 40, 62, 0.9) !important;
  border-color: rgba(99, 132, 255, 0.25) !important;
  color: #c7d2fe !important;
}

/* Text areas (SQL query, etc.) */
.stTextArea textarea,
[data-baseweb="textarea"] textarea,
[data-testid="stTextArea"] textarea {
  background: rgba(22, 30, 48, 0.95) !important;
  color: #e8eeff !important;
  border: 1px solid rgba(99, 132, 255, 0.35) !important;
  border-radius: 12px !important;
  caret-color: #93c5fd !important;
}

/* Code / schema blocks */
[data-testid="stCodeBlock"],
pre[class],
div[data-testid="stMarkdownContainer"] pre {
  background: rgba(16, 22, 36, 0.95) !important;
  border: 1px solid rgba(99, 132, 255, 0.28) !important;
  border-radius: 12px !important;
  color: #e2e8f0 !important;
}
[data-testid="stCodeBlock"] code,
pre code {
  background: transparent !important;
  color: #e2e8f0 !important;
}

/* Widget labels — readable on dark bg */
.stSelectbox label,
.stSlider label,
.stFileUploader label,
.stNumberInput label,
.stTextArea label {
  color: #a8b8d6 !important;
  font-size: 11px !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
}

/* Body copy in Streamlit markdown (threshold summary, captions, etc.) */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  color: #c8d4e8 !important;
}

/* Radio */
[data-testid="stRadio"] label,
[data-baseweb="radio"] label {
  color: #c8d4e8 !important;
}

/* Progress bar */
.stProgress > div > div { background: #2563eb !important; }

/* Divider */
hr { border-color: rgba(56,189,248,0.14) !important; }

/* Alert boxes */
.stAlert { background: linear-gradient(180deg, rgba(20,26,42,0.84), rgba(20,26,42,0.62)) !important; border: 1px solid var(--stroke) !important; border-radius: var(--radius); backdrop-filter: blur(12px); box-shadow: var(--shadow2); }

/* File uploader */
[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, rgba(20,26,42,0.78), rgba(20,26,42,0.62));
    border: 1px dashed rgba(56,189,248,0.22);
    border-radius: var(--radius);
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow2);
}

/* Reusable glass / 3D card for custom HTML blocks */
.glass-card{
  background: linear-gradient(180deg, rgba(20,26,42,0.86), rgba(20,26,42,0.66));
  border: 1px solid rgba(56,189,248,0.18);
  border-radius: var(--radius);
  padding: 16px;
  box-shadow: var(--shadow2);
  backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
  transform: translateZ(0);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.glass-card::before{
  content:"";
  position:absolute; inset:-1px;
  background:
    radial-gradient(240px 140px at 14% 18%, rgba(59,130,246,0.18), transparent 55%),
    radial-gradient(280px 180px at 86% 22%, rgba(167,139,250,0.16), transparent 60%);
  opacity:0.55;
  pointer-events:none;
}
.glass-card:hover{
  transform: perspective(900px) rotateX(4deg) rotateY(-6deg) translateY(-3px);
  box-shadow: var(--shadow);
  border-color: rgba(37,99,235,0.30);
}

/* Hero header */
.hero-shell{
  position: relative;
  padding: 18px 18px 14px 18px;
  margin: 18px 0 22px 0;
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(20,26,42,0.88), rgba(20,26,42,0.66));
  border: 1px solid rgba(56,189,248,0.18);
  box-shadow: 0 20px 60px rgba(2,6,23,0.52);
  overflow: hidden;
}
.hero-glow{
  position:absolute;
  inset:-2px;
  background:
    radial-gradient(540px 240px at 18% 0%, rgba(59,130,246,0.20), transparent 55%),
    radial-gradient(520px 260px at 80% 20%, rgba(167,139,250,0.20), transparent 58%),
    radial-gradient(520px 280px at 50% 120%, rgba(56,189,248,0.14), transparent 60%);
  filter: blur(10px);
  opacity: 0.9;
  pointer-events:none;
}
.hero-row{
  position: relative;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:16px;
}
.hero-brand{
  display:flex;
  align-items:center;
  gap:14px;
  min-width:280px;
}
.hero-dot{
  width:12px;
  height:12px;
  border-radius:50%;
  background:#2563eb;
  box-shadow: 0 0 14px rgba(37,99,235,0.55), 0 0 34px rgba(37,99,235,0.20);
  animation:pulse 2s infinite;
}
.hero-text{
  display:flex;
  flex-direction:column;
  line-height:1;
}
.hero-title{
  font-family:'Syne',sans-serif;
  font-size:30px;
  font-weight:800;
  letter-spacing:-1px;
  background: linear-gradient(90deg, #f8fbff 0%, #dbe8ff 40%, #60a5fa 100%);
  -webkit-background-clip:text;
  background-clip:text;
  color: transparent;
}
.hero-sub{
  font-size:11px;
  color:#9fb0d3;
  letter-spacing:2px;
  margin-top:8px;
  text-transform:uppercase;
}
.hero-chip{
  display:flex;
  align-items:center;
  gap:10px;
  padding:10px 12px;
  border-radius:999px;
  background: rgba(15, 20, 34, 0.58);
  border: 1px solid rgba(37,99,235,0.18);
  backdrop-filter: blur(10px);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
}
.hero-orb{
  width:36px;
  height:36px;
  border-radius:14px;
  background: radial-gradient(circle at 30% 30%, rgba(37,99,235,0.48), rgba(14,165,233,0.18) 55%, rgba(167,139,250,0.12) 78%, rgba(0,0,0,0) 100%);
  box-shadow: 0 18px 50px rgba(37,99,235,0.20);
  transform: perspective(800px) rotateX(14deg) rotateY(-18deg);
}
.hero-chip-meta{
  display:flex;
  flex-direction:column;
  gap:2px;
}
.hero-chip-label{
  font-size:10px;
  color:#4a6fa5;
  letter-spacing:1.5px;
  text-transform:uppercase;
}
.hero-chip-value{
  font-size:12px;
  color:#dbe8ff;
}

/* React-style section with layered mockup */
.mock-section{
  position: relative;
  margin: 18px 0 28px 0;
  padding: 34px 26px;
  border-radius: 28px;
  background: linear-gradient(180deg, rgba(20,26,42,0.88), rgba(20,26,42,0.64));
  border: 1px solid rgba(99,132,255,0.20);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.mock-section::before{
  content:"";
  position:absolute;
  inset:-1px;
  background:
    radial-gradient(560px 260px at 10% 0%, rgba(59,130,246,0.16), transparent 55%),
    radial-gradient(520px 220px at 90% 20%, rgba(168,85,247,0.14), transparent 56%);
  pointer-events:none;
}
.mock-grid{
  position: relative;
  z-index: 2;
  display: grid;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  gap: 24px;
}
.mock-grid.reverse .mock-text{ order: 2; }
.mock-grid.reverse .mock-visual{ order: 1; }
.mock-kicker{
  font-size: 11px;
  letter-spacing: 2px;
  color: #8fa6cf;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.mock-title{
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: clamp(26px, 3.2vw, 42px);
  line-height: 1.1;
  color: #f8fbff;
  margin: 0 0 12px 0;
}
.mock-desc{
  color: #a8b8d6;
  font-size: 14px;
  line-height: 1.9;
  max-width: 520px;
}
.mock-visual{
  position: relative;
  width: min(100%, 460px);
  margin-left: auto;
  margin-right: auto;
  height: 450px;
}
.mock-bg{
  position:absolute;
  width: 88%;
  height: 88%;
  left: -6%;
  top: 10%;
  border-radius: 28px;
  background-size: cover;
  background-position: center;
  filter: blur(1px) brightness(0.72);
  opacity: .82;
  transform: translateY(10px);
}
.mock-card{
  position:absolute;
  inset: 0 0 0 14%;
  border-radius: 28px;
  border: 1px solid rgba(147,180,255,0.22);
  background: rgba(255,255,255,0.04);
  backdrop-filter: blur(15px);
  overflow: hidden;
  box-shadow: 0 28px 60px rgba(2,6,23,0.52);
  animation: floatCard 6.5s ease-in-out infinite;
}
.mock-card-inner{
  width: 100%;
  height: 100%;
  background-size: cover;
  background-position: center;
}
@keyframes floatCard{
  0%,100%{ transform: translateY(0px) rotateX(0deg); }
  50%{ transform: translateY(-9px) rotateX(1.3deg); }
}
@media (max-width: 980px){
  .mock-grid{ grid-template-columns: 1fr; }
  .mock-grid.reverse .mock-text, .mock-grid.reverse .mock-visual{ order: initial; }
  .mock-visual{ height: 370px; width: min(100%, 360px); }
  .mock-section{ padding: 26px 18px; }
}

/* Links */
a, a:visited{ color: #2563eb; }
a:hover{ color: #0ea5e9; }

/* Force old inline dark cards to match the light glass palette */
div[style*="background:#0d1421"]{
  background: linear-gradient(180deg, rgba(20,26,42,0.86), rgba(20,26,42,0.66)) !important;
  border-color: rgba(99,132,255,0.24) !important;
  box-shadow: var(--shadow2) !important;
  color: #e6ecff !important;
}
div[style*="color:#c8d6e5"], span[style*="color:#c8d6e5"]{ color:#e6ecff !important; }
div[style*="color:#4a6fa5"], span[style*="color:#4a6fa5"]{ color:#9fb0d3 !important; }

/* Reduce motion accessibility */
@media (prefers-reduced-motion: reduce){
  .stApp::before{ animation: none !important; }
  *{ transition: none !important; }
}
</style>
""", unsafe_allow_html=True)

COLORS = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981",
          "lr": "#3b82f6", "rf": "#8b5cf6", "xgb": "#06b6d4", "accent": "#2563eb"}


def plotly_dark_table(
    df: pd.DataFrame,
    *,
    height: Optional[int] = None,
    max_rows: Optional[int] = None,
    include_index: bool = False,
) -> None:
    """Show a table with the same dark / transparent look as Plotly charts (avoids white Streamlit grid)."""
    d = df.copy()
    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows)
    if include_index:
        d = d.reset_index()
    else:
        d = d.reset_index(drop=True)
    for c in d.columns:
        d[c] = d[c].map(lambda x: "" if pd.isna(x) else str(x))
    n = len(d)
    ncols = len(d.columns)
    row_bg = [
        "rgba(26, 34, 54, 0.96)" if i % 2 == 0 else "rgba(20, 28, 46, 0.96)"
        for i in range(n)
    ]
    fill_per_col = [[row_bg[i] for i in range(n)] for _ in range(ncols)]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[str(c) for c in d.columns],
                    fill_color="rgba(32, 42, 68, 0.98)",
                    font=dict(color="#e8eeff", size=14),
                    align="left",
                    height=40,
                ),
                cells=dict(
                    values=[d[c].tolist() for c in d.columns],
                    fill_color=fill_per_col,
                    font=dict(color="#d8e2f0", size=13),
                    align="left",
                    height=30,
                ),
            )
        ]
    )
    h = height if height is not None else min(560, 70 + max(n, 1) * 32)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0),
        height=h,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": False},
    )


# ── Helpers ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_default_data():
    URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    try:
        return pd.read_csv(URL)
    except Exception:
        return None


@st.cache_resource
def train_models(df):
    df_model = df.copy()
    df_model, label_encoders = fit_label_encoders(df_model)

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

    # Permutation importance (model-agnostic explainability; uses RF on test subset for speed)
    n_perm = min(800, len(X_test))
    X_perm = X_test.iloc[:n_perm]
    y_perm = y_test.iloc[:n_perm]
    perm = permutation_importance(
        rf, X_perm, y_perm, n_repeats=8, random_state=42, n_jobs=-1
    )
    perm_importance = pd.Series(
        perm.importances_mean, index=X.columns
    ).sort_values(ascending=False)

    return (
        results,
        pred_df,
        feature_importance,
        best_name,
        X_test,
        y_test,
        scaler,
        X.columns.tolist(),
        label_encoders,
        perm_importance,
    )


def risk_color(score):
    if score >= 0.65: return COLORS["high"]
    if score >= 0.35: return COLORS["medium"]
    return COLORS["low"]

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-shell">
  <div class="hero-glow"></div>
  <div class="hero-row">
    <div class="hero-brand">
      <div class="hero-dot"></div>
      <div class="hero-text">
        <div class="hero-title">ChurnIQ</div>
        <div class="hero-sub">churn prediction · retention intelligence</div>
      </div>
    </div>
    <div class="hero-chip">
      <div class="hero-orb"></div>
      <div class="hero-chip-meta">
        <div class="hero-chip-label">dataset</div>
        <div class="hero-chip-value">IBM Telco · scikit-learn</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Dataset")
    data_source = st.radio(
        "Data source",
        ["Use IBM Telco Dataset (auto)", "Upload my own CSV"],
        label_visibility="collapsed",
    )
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
<div style="font-size:10px;color:#6b8aaf;line-height:1.8;">
ChurnIQ · IBM Telco · scikit-learn · Streamlit<br><br>
<a href="https://github.com/sadiquemahir" target="_blank" rel="noopener noreferrer"
   style="color:#00ff94;text-decoration:none;">github.com/sadiquemahir</a>
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
    (
        results,
        pred_df,
        feat_imp,
        best_name,
        X_test,
        y_test,
        scaler,
        feature_cols,
        label_encoders,
        perm_imp,
    ) = train_models(df)

best = results[best_name]

# ── TABS ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔍 EDA", "🤖 Model Comparison", 
    "🎯 Predictions", "🔮 Predict a Customer", "🗄️ SQL Explorer"
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    so_what_img, so_what_txt = st.columns([1, 1.12], gap="large")
    with so_what_img:
        st.markdown(
            """
            <div style="border-radius:18px;overflow:hidden;border:1px solid rgba(99,132,255,0.25);
            box-shadow:0 20px 50px rgba(2,6,23,0.5);">
            <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=900&q=80"
                 alt="Team reviewing analytics dashboards"
                 style="width:100%;display:block;vertical-align:middle;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Team analytics & dashboards — a visual fit for churn and retention decisions.")
    with so_what_txt:
        st.markdown(
            """
            <div style="color:#ffffff;">
            <h2 style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;color:#ffffff !important;margin:0 0 18px 0;">Why ChurnIQ?</h2>
            <div style="color:#ffffff;line-height:1.85;font-size:15px;">
            <p style="margin:0 0 16px 0;color:#ffffff !important;">
            <strong style="color:#ffffff;">Churn</strong> — Customers who cancel or stop paying. Catching risk early protects revenue and gives teams time to retain them.
            </p>
            <p style="margin:0 0 16px 0;color:#ffffff !important;">
            <strong style="color:#ffffff;">What ChurnIQ does</strong> — Trains on IBM Telco-style data (or your CSV), compares models (logistic regression, random forest, XGBoost when installed), and scores who is most likely to churn so you can prioritize outreach and retention.
            </p>
            <p style="margin:0;color:#ffffff !important;">
            <strong style="color:#ffffff;">How to use it</strong> — Load data in the sidebar → Overview for risk split and best model → EDA / Predictions for patterns and at-risk customers → Predict a Customer for what-if profiles → SQL Explorer when your database is connected.
            </p>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

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
        wedge_colors = [{"High": COLORS["high"], "Medium": COLORS["medium"], "Low": COLORS["low"]}[l] for l in risk_counts.index]
        risk_fig = go.Figure(
            data=[
                go.Pie(
                    labels=risk_counts.index.tolist(),
                    values=risk_counts.values.tolist(),
                    hole=0.0,
                    pull=[0.0 for _ in risk_counts.index],
                    marker=dict(colors=wedge_colors, line=dict(color="rgba(230,236,255,0.18)", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=18, color="#f8fbff"),
                    hovertemplate="<b>%{label}</b><br>Customers: %{value}<br>Share: %{percent}<extra></extra>",
                    hoverlabel=dict(
                        bgcolor="rgba(18,24,40,0.96)",
                        bordercolor="#3b82f6",
                        font=dict(size=15, color="#f8fbff")
                    ),
                )
            ]
        )
        risk_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            font=dict(color="#e6ecff"),
            showlegend=False,
        )
        st.plotly_chart(
            risk_fig,
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
        )

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
    c_a, c_b = st.columns(2)
    with c_a:
        if "MonthlyCharges" in df.columns:
            monthly_fig = go.Figure()
            monthly_fig.add_trace(go.Histogram(
                x=df[df["Churn"] == 0]["MonthlyCharges"],
                nbinsx=16, marker_color="#22c55e", opacity=0.92, name="Stayed",
                hovertemplate="Monthly: $%{x:.2f}<br>Customers: %{y}<extra>Stayed</extra>"
            ))
            monthly_fig.add_trace(go.Histogram(
                x=df[df["Churn"] == 1]["MonthlyCharges"],
                nbinsx=16, marker_color="#f97316", opacity=0.92, name="Churned",
                hovertemplate="Monthly: $%{x:.2f}<br>Customers: %{y}<extra>Churned</extra>"
            ))
            monthly_fig.update_layout(
                title=dict(text="Monthly Charges Distribution", x=0.5, xanchor="center", y=0.97, yref="paper", yanchor="top", font=dict(size=16, color="#e6ecff")),
                barmode="group",
                bargap=0.18,
                height=430,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6ecff", size=15),
                legend=dict(
                    orientation="h",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    xanchor="center",
                    y=0.88,
                    yanchor="bottom",
                    font=dict(color="#e8eeff", size=13),
                ),
                margin=dict(t=88, b=12, l=12, r=12),
                hovermode="closest",
                yaxis=dict(domain=[0.0, 0.74]),
            )
            monthly_fig.update_xaxes(title="Monthly Charges ($)", tickfont=dict(size=14), showgrid=False)
            monthly_fig.update_yaxes(title="Customers", tickfont=dict(size=14), gridcolor="rgba(159,176,211,0.22)")
            st.plotly_chart(monthly_fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False})
            st.caption("Each bar compares customers who stayed vs churned in the same monthly-charge range.")

    with c_b:
        if "tenure" in df.columns:
            tenure_fig = go.Figure()
            tenure_fig.add_trace(go.Histogram(
                x=df[df["Churn"] == 0]["tenure"],
                nbinsx=16, marker_color="#22c55e", opacity=0.92, name="Stayed",
                hovertemplate="Tenure: %{x} months<br>Customers: %{y}<extra>Stayed</extra>"
            ))
            tenure_fig.add_trace(go.Histogram(
                x=df[df["Churn"] == 1]["tenure"],
                nbinsx=16, marker_color="#f97316", opacity=0.92, name="Churned",
                hovertemplate="Tenure: %{x} months<br>Customers: %{y}<extra>Churned</extra>"
            ))
            tenure_fig.update_layout(
                title=dict(text="Tenure Distribution", x=0.5, xanchor="center", y=0.97, yref="paper", yanchor="top", font=dict(size=16, color="#e6ecff")),
                barmode="group",
                bargap=0.18,
                height=430,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6ecff", size=15),
                legend=dict(
                    orientation="h",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    xanchor="center",
                    y=0.88,
                    yanchor="bottom",
                    font=dict(color="#e8eeff", size=13),
                ),
                margin=dict(t=88, b=12, l=12, r=12),
                hovermode="closest",
                yaxis=dict(domain=[0.0, 0.74]),
            )
            tenure_fig.update_xaxes(title="Tenure (months)", tickfont=dict(size=14), showgrid=False)
            tenure_fig.update_yaxes(title="Customers", tickfont=dict(size=14), gridcolor="rgba(159,176,211,0.22)")
            st.plotly_chart(tenure_fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False})
            st.caption("Lower tenure bands usually show more churn. Longer-tenure customers are generally more stable.")

    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr().round(2)
    corr_fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            zmin=-1, zmax=1,
            colorscale="RdYlGn",
            text=corr.values,
            texttemplate="%{text}",
            textfont={"size": 13},
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar=dict(title=dict(text="Corr", font=dict(size=12)), tickfont=dict(size=12)),
        )
    )
    corr_fig.update_layout(
        title="Correlation Matrix",
        height=520,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6ecff", size=14),
        margin=dict(t=58, b=12, l=12, r=12),
    )
    corr_fig.update_xaxes(tickfont=dict(size=13))
    corr_fig.update_yaxes(tickfont=dict(size=13))
    st.plotly_chart(corr_fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False})

    # Key stats
    st.markdown("---")
    st.markdown("## 📌 Key Insights")
    cols = st.columns(3)
    if "Contract" in df.columns:
        mtm = df[df["Contract"]=="Month-to-month"]["Churn"].mean()*100
        twyr = df[df["Contract"]=="Two year"]["Churn"].mean()*100
        cols[0].markdown(f"""<div class="glass-card">
        <div style="font-size:9px;color:#4a6fa5;letter-spacing:2px;">CONTRACT IMPACT</div>
        <div style="font-size:20px;color:#ff6b6b;font-weight:700;margin:8px 0;">{mtm:.0f}% vs {twyr:.0f}%</div>
        <div style="font-size:11px;color:#8aadcc;">Month-to-month vs 2-year churn</div></div>""", unsafe_allow_html=True)
    if "tenure" in df.columns:
        avg_churned = df[df["Churn"]==1]["tenure"].mean()
        avg_retained = df[df["Churn"]==0]["tenure"].mean()
        cols[1].markdown(f"""<div class="glass-card">
        <div style="font-size:9px;color:#4a6fa5;letter-spacing:2px;">AVG TENURE</div>
        <div style="font-size:20px;color:#ffe66d;font-weight:700;margin:8px 0;">{avg_churned:.0f}mo vs {avg_retained:.0f}mo</div>
        <div style="font-size:11px;color:#8aadcc;">Churned vs retained customers</div></div>""", unsafe_allow_html=True)
    cols[2].markdown(f"""<div class="glass-card">
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
    plotly_dark_table(pd.DataFrame(metrics_data), height=260, include_index=False)

    st.markdown("### Threshold, precision / recall & cost framing (best model)")
    st.caption(
        "Move the threshold to trade off false alarms vs missed churners. "
        "Higher threshold → fewer positives (stricter churn alerts)."
    )
    thr_col1, thr_col2 = st.columns(2)
    with thr_col1:
        thr = st.slider("Probability threshold", 0.05, 0.95, 0.5, 0.05, key="thr_best")
    with thr_col2:
        cost_fn = st.number_input("Cost weight: missing a churn (FN)", min_value=1, max_value=50, value=5)
        cost_fp = st.number_input("Cost weight: false alarm (FP)", min_value=1, max_value=50, value=1)

    bp = best["probs"]
    pred_thr = (bp >= thr).astype(int)
    prec_t = precision_score(y_test, pred_thr, zero_division=0)
    rec_t = recall_score(y_test, pred_thr, zero_division=0)
    cm_thr = confusion_matrix(y_test, pred_thr)
    if cm_thr.size == 4:
        tn, fp, fn, tp = cm_thr.ravel()
    else:
        tn = fp = fn = tp = 0
    weighted_cost = fn * cost_fn + fp * cost_fp
    st.markdown(
        f"**At threshold {thr:.2f}** — precision **{prec_t:.3f}**, recall **{rec_t:.3f}** · "
        f"TP={tp}, FP={fp}, TN={tn}, FN={fn} · "
        f"weighted cost (FN×{cost_fn} + FP×{cost_fp}) = **{weighted_cost}**"
    )

    st.markdown("### Calibration (best model)")
    cal_y, cal_x = calibration_curve(y_test, bp, n_bins=10, strategy="uniform")
    cal_fig = go.Figure()
    cal_fig.add_trace(
        go.Scatter(
            x=cal_x, y=cal_y, mode="lines+markers",
            name="Model", line=dict(color="#60a5fa", width=2),
            marker=dict(size=8),
        )
    )
    cal_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                   line=dict(color="#94a3b8", dash="dash", width=1)))
    cal_fig.update_layout(
        title="Predicted probability vs observed churn rate",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6ecff"), margin=dict(t=48, b=12, l=12, r=12),
        height=360, xaxis_title="Mean predicted probability", yaxis_title="Fraction positives (churn)",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#e8eeff", size=12)),
    )
    cal_fig.update_xaxes(showgrid=True, gridcolor="rgba(159,176,211,0.16)")
    cal_fig.update_yaxes(showgrid=True, gridcolor="rgba(159,176,211,0.16)")
    st.plotly_chart(cal_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## ROC Curves")
        roc_fig = go.Figure()
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res["probs"])
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode="lines",
                    line=dict(color=res["color"], width=4 if name == best_name else 2),
                    name=f"{name} (AUC={res['auc']:.3f})",
                    hovertemplate=f"{name}<br>FPR: "+"%{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
                )
            )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color="#8fa6cf", dash="dash", width=1.5),
                name="Random",
                hoverinfo="skip",
            )
        )
        roc_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            font=dict(color="#e6ecff"),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            hovermode="closest",
            legend=dict(
                bgcolor="rgba(20,26,42,0.65)",
                bordercolor="rgba(159,176,211,0.2)",
                borderwidth=1,
                font=dict(color="#e8eeff", size=12),
            ),
        )
        roc_fig.update_xaxes(showgrid=True, gridcolor="rgba(159,176,211,0.16)")
        roc_fig.update_yaxes(showgrid=True, gridcolor="rgba(159,176,211,0.16)")
        st.plotly_chart(roc_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    with col2:
        st.markdown("## Feature Importance (Random Forest)")
        top12 = feat_imp.head(12).sort_values()
        colors_fi = [COLORS["high"] if v > top12.median() else COLORS["lr"] for v in top12.values]
        fi_fig = go.Figure(
            go.Bar(
                x=top12.values,
                y=top12.index,
                orientation="h",
                marker=dict(color=colors_fi),
                text=[f"{v:.3f}" for v in top12.values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        )
        fi_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            font=dict(color="#e6ecff"),
            xaxis_title="Importance Score",
            yaxis_title="",
        )
        fi_fig.update_xaxes(showgrid=True, gridcolor="rgba(159,176,211,0.16)")
        fi_fig.update_yaxes(showgrid=False)
        st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    st.markdown("---")
    st.markdown("## Confusion Matrix — Best Model")
    cm = confusion_matrix(y_test, best["preds"])
    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred Retained", "Pred Churned"],
            y=["Actual Retained", "Actual Churned"],
            colorscale="YlOrRd",
            text=cm,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
            showscale=False,
        )
    )
    cm_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(color="#e6ecff"),
    )
    st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    st.markdown("---")
    st.markdown("## Permutation importance (model-agnostic)")
    st.caption(
        "How much model accuracy drops when a feature is shuffled (on a test subset). "
        "Complements tree-based Gini importance."
    )
    top_perm = perm_imp.head(12).sort_values()
    perm_fig = go.Figure(
        go.Bar(
            x=top_perm.values,
            y=top_perm.index,
            orientation="h",
            marker=dict(color="#a78bfa"),
            text=[f"{v:.4f}" for v in top_perm.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Mean Δ accuracy: %{x:.5f}<extra></extra>",
        )
    )
    perm_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10), font=dict(color="#e6ecff"),
        xaxis_title="Mean decrease in accuracy", height=420,
    )
    perm_fig.update_xaxes(showgrid=True, gridcolor="rgba(159,176,211,0.16)")
    st.plotly_chart(perm_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    st.markdown("## SHAP values (Random Forest)")
    try:
        import shap
        rf_shap = results["Random Forest"]["model"]
        n_shap = min(400, len(X_test))
        X_shap = X_test.iloc[:n_shap]
        explainer = shap.TreeExplainer(rf_shap)
        sv = explainer.shap_values(X_shap)
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            # Binary class: shape (n_samples, n_features, 2) — use positive class
            if sv.shape[-1] == 2:
                sv = sv[..., 1]
            elif sv.shape[1] == 2:
                sv = sv[:, 1, :]
            elif sv.shape[0] == 2:
                sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        mean_abs = np.asarray(mean_abs).ravel()
        shap_ser = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False).head(12)
        shap_fig = go.Figure(
            go.Bar(
                x=shap_ser.values,
                y=shap_ser.index,
                orientation="h",
                marker=dict(color="#38bdf8"),
                text=[f"{v:.4f}" for v in shap_ser.values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
            )
        )
        shap_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10), font=dict(color="#e6ecff"),
            xaxis_title="Mean |SHAP|", height=420,
        )
        shap_fig.update_xaxes(showgrid=True, gridcolor="rgba(159,176,211,0.16)")
        st.plotly_chart(shap_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
        st.caption(f"Computed on {n_shap} test rows. Larger values push the model more toward churn prediction.")
    except ImportError:
        st.info("Install **shap** for SHAP explanations: `pip install shap`")
    except Exception as e:
        st.warning(f"SHAP could not be computed: {e}")


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
        bar_colors = [COLORS["high"] if r=="High" else COLORS["medium"] for r in top20["Risk_Level"]]
        risk_fig = go.Figure(
            go.Bar(
                x=(top20["Churn_Probability"] * 100).round(2),
                y=[f"Customer {i}" for i in top20.index],
                orientation="h",
                marker=dict(color=bar_colors),
                text=[f"{p*100:.1f}%" for p in top20["Churn_Probability"]],
                textposition="outside",
                customdata=top20["Risk_Level"],
                hovertemplate="<b>%{y}</b><br>Risk: %{customdata}<br>Probability: %{x:.1f}%<extra></extra>",
            )
        )
        risk_fig.add_vline(x=65, line_dash="dash", line_color=COLORS["medium"], opacity=0.85)
        risk_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=20),
            font=dict(color="#e6ecff"),
            xaxis_title="Churn Probability (%)",
            yaxis_title="",
            xaxis=dict(range=[0, 108], showgrid=True, gridcolor="rgba(159,176,211,0.18)"),
            yaxis=dict(autorange="reversed"),
            hovermode="closest",
            showlegend=False,
        )
        st.plotly_chart(risk_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    with col_table:
        st.markdown("## All Predictions")
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
        display_df = pred_df[["Churn_Probability", "Risk_Level", "Predicted_Churn", "Actual_Churn"]].copy()
        display_df["Churn_Probability"] = (display_df["Churn_Probability"] * 100).round(1).astype(str) + "%"
        if risk_filter != "All":
            display_df = display_df[display_df["Risk_Level"] == risk_filter]
        plotly_dark_table(display_df, height=440, include_index=True)

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

            row_dict = sample.iloc[0].to_dict()
            sample_encoded = encode_row_with_encoders(row_dict, label_encoders, feature_cols)

            rf_model = results["Random Forest"]["model"]
            prob = rf_model.predict_proba(sample_encoded)[0][1]
            risk = "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low"
            rcolor = COLORS["high"] if risk=="High" else COLORS["medium"] if risk=="Medium" else COLORS["low"]

            st.markdown(f"""
            <div class="glass-card" style="border-color:{rcolor}; padding:32px; text-align:center; margin-top:16px;">
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


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — SQL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 🗄️ SQL Explorer")
    if _get_engine() is None:
        st.info(
            "Connect a database by setting **DATABASE_URL** (environment variable or "
            "`.streamlit/secrets.toml`). See **README.md**."
        )
    st.markdown("""
    <div style='font-size:12px;color:#4a6fa5;margin-bottom:24px;'>
    Query your churn database directly using SQL.
    </div>
    """, unsafe_allow_html=True)

    # Preset queries
    st.markdown("### Quick Queries")
    preset = st.selectbox("Choose a query or write your own below:", [
        "Show all high risk customers",
        "Average churn probability by risk level",
        "Count of predicted vs actual churn",
        "Top 10 highest risk customers",
        "Custom query"
    ])

    queries = {
        "Show all high risk customers": 
            "SELECT * FROM churn_predictions WHERE risk_level = 'High' LIMIT 50;",
        "Average churn probability by risk level": 
            "SELECT risk_level, ROUND(AVG(churn_probability)::numeric, 3) as avg_probability, COUNT(*) as total_customers FROM churn_predictions GROUP BY risk_level ORDER BY avg_probability DESC;",
        "Count of predicted vs actual churn": 
            "SELECT predicted_churn, actual_churn, COUNT(*) as count FROM churn_predictions GROUP BY predicted_churn, actual_churn ORDER BY predicted_churn;",
        "Top 10 highest risk customers": 
            "SELECT customer_id, ROUND(churn_probability::numeric, 3) as churn_probability, risk_level FROM churn_predictions ORDER BY churn_probability DESC LIMIT 10;",
        "Custom query": ""
    }

    # Show query in text area
    query = st.text_area(
        "SQL Query:", 
        value=queries[preset],
        height=120
    )

    if st.button("▶ RUN QUERY"):
        if query.strip():
            try:
                result = load_from_db(query)
                st.success(f"✅ Returned {len(result):,} rows")
                if len(result) > 500:
                    st.caption("Preview shows the first **500** rows. Use download for the full result set.")
                plotly_dark_table(result, height=480, max_rows=500, include_index=False)

                # Download results
                csv = result.to_csv(index=False)
                st.download_button(
                    "⬇ Download Results CSV",
                    csv,
                    "query_results.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"❌ Query error: {e}")
        else:
            st.warning("Please enter a SQL query")

    # Schema reference
    st.markdown("---")
    st.markdown("### 📋 Table Schema")
    st.code("""
    TABLE: churn_predictions
    ┌─────────────────────┬─────────────┐
    │ Column              │ Type        │
    ├─────────────────────┼─────────────┤
    │ customer_id         │ INTEGER     │
    │ churn_probability   │ FLOAT       │
    │ risk_level          │ VARCHAR(10) │
    │ predicted_churn     │ INTEGER     │
    │ actual_churn        │ INTEGER     │
    └─────────────────────┴─────────────┘
    """)
                


