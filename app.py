import streamlit as st
import pandas as pd
from model_utils import predict_subtype, predict_batch, get_feature_columns

st.set_page_config(page_title="ClinSTRAT", page_icon="🧬", layout="centered")
st.title("🧬 ClinSTRAT — Breast Cancer Subtype Classifier")
st.caption("Multimodal (clinical + gene expression + pathway) XGBoost model, 6-class prediction.")

mode = st.radio(
    "Input mode",
    ["Upload CSV (recommended — full clinical + genomic row)", "Manual clinical-only entry (demo)"],
)

st.divider()

if mode.startswith("Upload"):
    st.write(
        "Upload a CSV with one row per patient, using the same raw columns as "
        "`test_patients.csv` / `feature_matrix.csv` (clinical fields + gene expression + pathway scores). "
        "Missing columns are filled with 0."
    )
    st.caption(
        "⚠️ This model needs a real microarray gene expression profile per patient — not just clinical "
        "symptoms. It's built for labs/researchers with existing sequencing data, not as a walk-up symptom checker."
    )

    template_cols = ["PATIENT_ID", "CLAUDIN_SUBTYPE"] + get_feature_columns()
    template_csv = pd.DataFrame(columns=template_cols).to_csv(index=False)
    st.download_button(
        "Download a blank template CSV (correct column headers)",
        template_csv,
        file_name="clinstrat_template.csv",
    )

    file = st.file_uploader("Patient data CSV", type=["csv"])
    if file is not None:
        try:
            df_raw = pd.read_csv(file)
        except Exception:
            st.error("Couldn't read that file — make sure it's a valid, comma-separated CSV.")
            st.stop()

        if df_raw.empty or df_raw.shape[1] == 0:
            st.error("This CSV is empty. Upload a file with at least one patient row.")
            st.stop()

        expected_cols = set(get_feature_columns())
        overlap = expected_cols.intersection(df_raw.columns)
        if len(overlap) < 5:
            st.error(
                f"This file only matches {len(overlap)} of the ~{len(expected_cols)} expected columns — "
                "it doesn't look like it's in the required format. Download the template above, or reuse "
                "`test_patients.csv` / `feature_matrix.csv`, and match those column headers."
            )
            st.stop()

        true_labels = df_raw["CLAUDIN_SUBTYPE"].copy() if "CLAUDIN_SUBTYPE" in df_raw.columns else None
        drop_cols = [c for c in ["PATIENT_ID", "CLAUDIN_SUBTYPE"] if c in df_raw.columns]
        df_raw = df_raw.drop(columns=drop_cols)

        if st.button("Predict"):
            try:
                with st.spinner("Running model..."):
                    results = predict_batch(df_raw)
            except Exception as e:
                st.error(f"Prediction failed — the file's columns don't line up with what the model expects. ({e})")
                st.stop()
            st.success(f"Predicted {len(results)} patient(s).")

            if true_labels is not None:
                from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

                mask = true_labels.notna() & (true_labels != "NC")
                y_true = true_labels[mask].reset_index(drop=True)
                y_pred = results.loc[mask.values, "predicted_subtype"].reset_index(drop=True)

                acc = accuracy_score(y_true, y_pred)
                f1_macro = f1_score(y_true, y_pred, average="macro")
                kappa = cohen_kappa_score(y_true, y_pred)

                st.subheader("Accuracy check (CLAUDIN_SUBTYPE column found)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{acc:.1%}")
                c2.metric("Macro F1", f"{f1_macro:.3f}")
                c3.metric("Cohen's Kappa", f"{kappa:.3f}")

                labels_sorted = sorted(y_true.unique())
                cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
                cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
                st.write("Confusion matrix (rows = actual, columns = predicted)")
                st.dataframe(cm_df)

            st.dataframe(results)
            st.download_button(
                "Download predictions CSV",
                results.to_csv(index=False),
                file_name="clinstrat_predictions.csv",
            )

else:
    st.warning(
        "Demo mode: only clinical fields are set. All ~550 gene/pathway features "
        "default to 0, which will bias predictions — use CSV upload for real accuracy."
    )
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age at diagnosis", 20, 100, 60)
        npi = st.number_input("NPI", 0.0, 10.0, 5.0, step=0.1)
        lymph_nodes = st.number_input("Lymph nodes examined positive", 0, 50, 0)
        cellularity = st.selectbox("Cellularity", ["High", "Moderate", "Low"])
    with col2:
        chemo = st.selectbox("Chemotherapy", ["YES", "NO"])
        hormone = st.selectbox("Hormone therapy", ["YES", "NO"])
        er_ihc = st.selectbox("ER IHC status", ["Positve", "Negative"])  # matches source data spelling
        radio = st.selectbox("Radiotherapy", ["YES", "NO"])

    if st.button("Predict"):
        patient = {
            "AGE_AT_DIAGNOSIS": age,
            "NPI": npi,
            "LYMPH_NODES_EXAMINED_POSITIVE": lymph_nodes,
            "CELLULARITY": cellularity,
            "CHEMOTHERAPY": chemo,
            "HORMONE_THERAPY": hormone,
            "ER_IHC": er_ihc,
            "RADIO_THERAPY": radio,
        }
        label, probs = predict_subtype(patient)
        st.subheader(f"Predicted subtype: `{label}`")
        st.bar_chart(pd.Series(probs))

with st.expander("Model info"):
    st.write(f"Feature count: {len(get_feature_columns())}")
    st.write("Classes: Basal, Her2, LumA, LumB, Normal, claudin-low")
