import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/crop_yield_model.joblib")
METRICS_PATH = Path("models/model_metrics.json")
DATA_PATH = Path("data/processed/cleaned_crop_yield_data.csv")


FEATURE_LABELS = {
    "Crop": "Crop",
    "Region": "Region",
    "Soil_Type": "Soil type",
    "Soil_pH": "Soil pH",
    "Rainfall_mm": "Rainfall (mm)",
    "Temperature_C": "Temperature (C)",
    "Humidity_pct": "Humidity (%)",
    "Fertilizer_Used_kg": "Fertilizer used (kg)",
    "Irrigation": "Irrigation method",
    "Pesticides_Used_kg": "Pesticides used (kg)",
    "Planting_Density": "Planting density",
    "Previous_Crop": "Previous crop",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_default_value(df: pd.DataFrame, column: str) -> float:
    return float(df[column].median())


st.set_page_config(page_title="Smart Crop Yield Prediction", layout="wide")
st.title("Smart Crop Yield Prediction System")
st.caption("Yield prediction using Kaggle crop, soil, weather, and farm management data")

if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not DATA_PATH.exists():
    st.error("Project files are missing. Run `python src/train_model.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()
reference_df = load_reference_data()
feature_columns = metrics["feature_columns"]
categorical_features = metrics.get(
    "categorical_features",
    reference_df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist(),
)

left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("Input Farm Conditions")
    inputs = {}

    for column in feature_columns:
        label = FEATURE_LABELS.get(column, column)

        if column in categorical_features:
            options = sorted(reference_df[column].dropna().astype(str).unique())
            inputs[column] = st.selectbox(label, options)
        else:
            default = get_default_value(reference_df, column)
            minimum = float(reference_df[column].min())
            maximum = float(reference_df[column].max())
            inputs[column] = st.slider(label, minimum, maximum, default)

    input_df = pd.DataFrame([inputs], columns=feature_columns)

    if st.button("Predict Crop Yield", type="primary"):
        prediction = float(model.predict(input_df)[0])
        st.success(f"Predicted crop yield: {prediction:.2f} tons per hectare")

with right:
    st.subheader("Model Test Results")
    st.metric("R2 Score", f"{metrics['r2_score']:.2%}")
    st.metric("Mean Absolute Error", f"{metrics['mean_absolute_error']:.2f} tons/ha")
    st.metric("RMSE", f"{metrics['root_mean_squared_error']:.2f} tons/ha")

    st.write("Dataset Preview")
    st.dataframe(reference_df.head(10), use_container_width=True)
