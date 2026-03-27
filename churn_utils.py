"""Shared data prep and encoding helpers (used by app + tests)."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    med = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(med)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["Churn"].isnull().any():
        df["Churn"] = df["Churn"].fillna(0).astype(int)
    if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype in [int, float]:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
    return df


def fit_label_encoders(df_model: pd.DataFrame) -> tuple:
    """Fit one LabelEncoder per object column; return encoded frame and encoders dict."""
    df_model = df_model.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in df_model.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le
    return df_model, encoders


def encode_row_with_encoders(
    row: dict,
    encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Encode a single observation using training encoders (unknown categories → first class)."""
    out: dict = {}
    for col in feature_cols:
        if col in encoders:
            le = encoders[col]
            val = str(row[col])
            if val not in le.classes_:
                val = le.classes_[0]
            out[col] = le.transform([val])[0]
        else:
            out[col] = row[col]
    return pd.DataFrame([out])[feature_cols]
