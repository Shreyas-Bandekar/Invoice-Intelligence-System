from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Invoice Intelligence System", page_icon="📊", layout="wide")

ROOT_DIR = Path(__file__).resolve().parent
FREIGHT_MODEL_PATH = ROOT_DIR / "Freight Cost Prediction" / "models" / "predict_freight_model.pkl"
INVOICE_MODEL_PATH = ROOT_DIR / "Invoice Flagging" / "models" / "invoice_flagging_model.pkl"


@st.cache_resource
def load_freight_model():
    return joblib.load(FREIGHT_MODEL_PATH)


@st.cache_resource
def load_invoice_bundle():
    return joblib.load(INVOICE_MODEL_PATH)


st.title("Invoice Intelligence System")
st.caption("Freight prediction + invoice risk flagging")

mode = st.sidebar.radio(
    "Select module",
    ["Freight Cost Prediction", "Invoice Flagging"],
)

if mode == "Freight Cost Prediction":
    st.subheader("Predict Freight Cost")

    if not FREIGHT_MODEL_PATH.exists():
        st.error("Freight model not found. Run Freight Cost Prediction/train.py first.")
    else:
        model = load_freight_model()

        col1, col2 = st.columns(2)
        with col1:
            dollars = st.number_input("Invoice dollars", min_value=0.0, value=1000.0, step=50.0)
        with col2:
            quantity = st.number_input("Quantity (optional context)", min_value=0.0, value=100.0, step=1.0)

        if st.button("Predict freight", use_container_width=True):
            input_df = pd.DataFrame({"Dollars": [dollars]})
            pred = float(model.predict(input_df)[0])
            st.success(f"Predicted freight cost: {pred:.2f}")
            st.info("Current freight model is trained using the Dollars feature only.")

else:
    st.subheader("Flag Invoice Risk")

    if not INVOICE_MODEL_PATH.exists():
        st.error("Invoice flagging model not found. Run Invoice Flagging/train.py first.")
    else:
        bundle = load_invoice_bundle()
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]

        defaults = {
            "invoice_quantity": 100.0,
            "invoice_dollars": 1000.0,
            "Freight": 120.0,
            "total_brands": 5.0,
            "total_item_quantity": 120.0,
            "days_po_to_invoice": 7.0,
            "total_item_dollars": 980.0,
        }

        inputs = {}
        cols = st.columns(2)
        for idx, feature in enumerate(feature_columns):
            with cols[idx % 2]:
                inputs[feature] = st.number_input(
                    feature,
                    value=float(defaults.get(feature, 0.0)),
                    step=1.0,
                )

        if st.button("Predict flag", use_container_width=True):
            input_df = pd.DataFrame([inputs], columns=feature_columns)
            pred = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                risk_prob = float(model.predict_proba(input_df)[0][1])
                st.metric("Flag risk probability", f"{risk_prob:.2%}")

            if pred == 1:
                st.error("Flag for manual review")
            else:
                st.success("Low risk: auto-approval candidate")

        st.divider()
        st.markdown("### Batch Scoring (CSV)")
        uploaded = st.file_uploader("Upload CSV with required feature columns", type=["csv"])

        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            missing_cols = [c for c in feature_columns if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                pred_df = batch_df.copy()
                pred_df["flag_prediction"] = model.predict(pred_df[feature_columns])
                if hasattr(model, "predict_proba"):
                    pred_df["flag_probability"] = model.predict_proba(pred_df[feature_columns])[:, 1]

                st.dataframe(pred_df.head(20), use_container_width=True)
                st.download_button(
                    "Download predictions",
                    data=pred_df.to_csv(index=False).encode("utf-8"),
                    file_name="invoice_flag_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
