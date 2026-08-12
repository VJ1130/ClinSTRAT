"""
ClinSTRAT model loading + prediction logic.
Extracted from 04_modeling.ipynb, cells 10-11.
"""
import os
import joblib
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_model = None
_scaler = None
_feature_columns = None
_label_encoder = None


def load_artifacts():
    """Load model/scaler/features/encoder once, cache in module-level vars."""
    global _model, _scaler, _feature_columns, _label_encoder
    if _model is None:
        _model = joblib.load(os.path.join(MODELS_DIR, "clinstrat_model.pkl"))
        _scaler = joblib.load(os.path.join(MODELS_DIR, "clinstrat_scaler.pkl"))
        _feature_columns = joblib.load(os.path.join(MODELS_DIR, "clinstrat_features.pkl"))
        _label_encoder = joblib.load(os.path.join(MODELS_DIR, "clinstrat_label_encoder.pkl"))
    return _model, _scaler, _feature_columns, _label_encoder


def get_feature_columns():
    _, _, feature_columns, _ = load_artifacts()
    return feature_columns


def predict_subtype(new_patient_dict):
    """
    Same logic as notebook 04, cell 11.
    new_patient_dict: dict or single-row DataFrame of raw (pre-encoding) feature values.
    Returns (predicted_label, {class_name: probability}).
    """
    model, scaler, feature_columns, label_encoder = load_artifacts()

    new_df = pd.DataFrame([new_patient_dict]) if isinstance(new_patient_dict, dict) else new_patient_dict.copy()

    cat_cols = new_df.select_dtypes(include=["object", "str"]).columns.tolist()
    new_df = pd.get_dummies(new_df, columns=cat_cols, dummy_na=True)
    new_df = new_df.reindex(columns=feature_columns, fill_value=0)

    new_scaled = scaler.transform(new_df)
    pred_encoded = model.predict(new_scaled)
    pred_proba = model.predict_proba(new_scaled)

    pred_label = label_encoder.inverse_transform(pred_encoded)[0]
    proba_dict = dict(zip(label_encoder.classes_, pred_proba[0]))
    return pred_label, proba_dict


def predict_batch(df_raw):
    """Run predict_subtype row-by-row on a DataFrame of raw feature rows. Returns a results DataFrame."""
    rows = []
    for _, row in df_raw.iterrows():
        label, probs = predict_subtype(row.to_dict())
        rows.append({"predicted_subtype": label, **{f"prob_{k}": v for k, v in probs.items()}})
    return pd.DataFrame(rows, index=df_raw.index)
