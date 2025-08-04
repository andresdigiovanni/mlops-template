from pathlib import Path

import requests
import streamlit as st

# FastAPI endpoint URL
API_URL = "http://api:8000/predict"

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🔬 Breast Cancer Prediction App")
st.markdown("Enter patient data to get prediction from trained model.")

# Split input into columns
col1, col2, col3 = st.columns(3)

# Define input fields
with col1:
    mean_radius = st.number_input("Mean Radius", 0.0, 50.0, 14.5)
    mean_texture = st.number_input("Mean Texture", 0.0, 50.0, 20.5)
    mean_perimeter = st.number_input("Mean Perimeter", 0.0, 200.0, 96.2)
    mean_area = st.number_input("Mean Area", 0.0, 3000.0, 644.1)
    mean_smoothness = st.number_input("Mean Smoothness", 0.0, 1.0, 0.105)
    mean_compactness = st.number_input("Mean Compactness", 0.0, 1.0, 0.15)
    mean_concavity = st.number_input("Mean Concavity", 0.0, 1.0, 0.12)
    mean_concave_points = st.number_input("Mean Concave Points", 0.0, 1.0, 0.075)
    mean_symmetry = st.number_input("Mean Symmetry", 0.0, 1.0, 0.18)
    mean_fractal_dimension = st.number_input("Mean Fractal Dimension", 0.0, 1.0, 0.06)

with col2:
    radius_error = st.number_input("Radius Error", 0.0, 10.0, 0.55)
    texture_error = st.number_input("Texture Error", 0.0, 10.0, 1.0)
    perimeter_error = st.number_input("Perimeter Error", 0.0, 10.0, 3.3)
    area_error = st.number_input("Area Error", 0.0, 200.0, 40.0)
    smoothness_error = st.number_input("Smoothness Error", 0.0, 1.0, 0.006)
    compactness_error = st.number_input("Compactness Error", 0.0, 1.0, 0.015)
    concavity_error = st.number_input("Concavity Error", 0.0, 1.0, 0.02)
    concave_points_error = st.number_input("Concave Points Error", 0.0, 1.0, 0.01)
    symmetry_error = st.number_input("Symmetry Error", 0.0, 1.0, 0.015)
    fractal_dimension_error = st.number_input(
        "Fractal Dimension Error", 0.0, 1.0, 0.003
    )

with col3:
    worst_radius = st.number_input("Worst Radius", 0.0, 50.0, 16.5)
    worst_texture = st.number_input("Worst Texture", 0.0, 50.0, 28.0)
    worst_perimeter = st.number_input("Worst Perimeter", 0.0, 200.0, 105.0)
    worst_area = st.number_input("Worst Area", 0.0, 3000.0, 800.0)
    worst_smoothness = st.number_input("Worst Smoothness", 0.0, 1.0, 0.13)
    worst_compactness = st.number_input("Worst Compactness", 0.0, 1.0, 0.2)
    worst_concavity = st.number_input("Worst Concavity", 0.0, 1.0, 0.25)
    worst_concave_points = st.number_input("Worst Concave Points", 0.0, 1.0, 0.15)
    worst_symmetry = st.number_input("Worst Symmetry", 0.0, 1.0, 0.25)
    worst_fractal_dimension = st.number_input("Worst Fractal Dimension", 0.0, 1.0, 0.08)

# Prediction button
if st.button("🧠 Predict"):
    input_payload = [
        {
            "mean_radius": mean_radius,
            "mean_texture": mean_texture,
            "mean_perimeter": mean_perimeter,
            "mean_area": mean_area,
            "mean_smoothness": mean_smoothness,
            "mean_compactness": mean_compactness,
            "mean_concavity": mean_concavity,
            "mean_concave_points": mean_concave_points,
            "mean_symmetry": mean_symmetry,
            "mean_fractal_dimension": mean_fractal_dimension,
            "radius_error": radius_error,
            "texture_error": texture_error,
            "perimeter_error": perimeter_error,
            "area_error": area_error,
            "smoothness_error": smoothness_error,
            "compactness_error": compactness_error,
            "concavity_error": concavity_error,
            "concave_points_error": concave_points_error,
            "symmetry_error": symmetry_error,
            "fractal_dimension_error": fractal_dimension_error,
            "worst_radius": worst_radius,
            "worst_texture": worst_texture,
            "worst_perimeter": worst_perimeter,
            "worst_area": worst_area,
            "worst_smoothness": worst_smoothness,
            "worst_compactness": worst_compactness,
            "worst_concavity": worst_concavity,
            "worst_concave_points": worst_concave_points,
            "worst_symmetry": worst_symmetry,
            "worst_fractal_dimension": worst_fractal_dimension,
        }
    ]

    try:
        response = requests.post(API_URL, json=input_payload)
        if response.status_code == 200:
            results = response.json()
            for res in results:
                st.success(
                    f"🎯 Prediction: `{res['label']}` | Probability: `{res['prediction_prob']:.4f}`"
                )
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.exception(f"API call failed: {e}")

# Visualize Drift
st.subheader("📊 Drift Reports")
drift_type = st.radio(
    "Select drift report to view:", ("Data Drift", "Prediction Drift")
)

drift_filename = "data_drift.html" if drift_type == "Data Drift" else "pred_drift.html"
drift_report_path = Path(".drift_reports") / drift_filename

if drift_report_path.exists():
    with st.expander(f"📉 {drift_type} Report"):
        st.components.v1.html(drift_report_path.read_text(), height=800, scrolling=True)
else:
    st.info(f"{drift_type} report not yet available.")
