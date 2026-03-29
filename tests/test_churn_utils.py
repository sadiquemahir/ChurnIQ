import pandas as pd

from churn_utils import clean_data, encode_row_with_encoders, fit_label_encoders


def test_clean_data_maps_churn_and_totalcharges():
    df = pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "Churn": ["Yes", "No"],
            "TotalCharges": ["12.5", "30"],
            "foo": [1, 2],
        }
    )
    out = clean_data(df)
    assert "customerID" not in out.columns
    assert out["Churn"].tolist() == [1, 0]
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])


def test_encode_row_with_encoders_roundtrip():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a"],
            "num": [1.0, 2.0, 3.0],
            "Churn": [0, 1, 0],
        }
    )
    df_enc, encoders = fit_label_encoders(df.drop(columns=["Churn"]))
    row = {"cat": "b", "num": 2.0}
    out = encode_row_with_encoders(row, encoders, ["cat", "num"])
    assert out.shape == (1, 2)
    assert out["num"].iloc[0] == 2.0


def test_unknown_category_falls_back():
    df = pd.DataFrame({"cat": ["x", "y"], "n": [1, 2]})
    df_enc, encoders = fit_label_encoders(df)
    row = {"cat": "never_seen", "n": 3}
    out = encode_row_with_encoders(row, encoders, ["cat", "n"])
    # Falls back to first class label
    assert out["cat"].iloc[0] == 0


def test_fit_label_encoders_preserves_numeric_columns():
    df = pd.DataFrame({"region": ["A", "B"], "amount": [10.5, 20.0]})
    encoded, encoders = fit_label_encoders(df)
    assert "region" in encoders
    assert "amount" not in encoders
    assert encoded["amount"].tolist() == [10.5, 20.0]
    assert encoded["region"].astype(int).tolist() == [0, 1]


def test_clean_data_coerces_numeric_churn_when_binary_strings():
    """Binary Churn as Yes/No is the primary path; NaN churn rows become 0."""
    df = pd.DataFrame({"Churn": ["Yes", "No"], "TotalCharges": [1.0, 2.0]})
    out = clean_data(df)
    assert out["Churn"].tolist() == [1, 0]
