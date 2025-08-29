from pathlib import Path
import requests
import streamlit as st

# FastAPI endpoint URL
API_URL = "http://api:8000/predict"

st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")
st.title("🏦 Bank Marketing Prediction App")
st.markdown("Enter client data to get prediction from the trained model.")

# Column layout
col1, col2, col3 = st.columns(3)

# Input fields
with col1:
    age = st.number_input("Age", 18, 90, 35)
    job = st.selectbox(
        "Job",
        [
            "management",
            "blue-collar",
            "unemployed",
            "housemaid",
            "technician",
            "retired",
            "admin.",
            "services",
            "self-employed",
        ],
    )
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox(
        "Education", ["primary", "secondary", "tertiary", "unknown"]
    )
    default = st.selectbox("Has Credit Default?", ["yes", "no"])
    balance = st.number_input("Balance", -1000, 100000, 1500)

with col2:
    housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Has Personal Loan?", ["yes", "no"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    day = st.number_input("Last Contact Day of Month", 1, 31, 15)
    month = st.selectbox(
        "Last Contact Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
    )
    duration = st.number_input("Last Contact Duration (seconds)", 0, 5000, 300)

with col3:
    campaign = st.number_input("Number of Contacts During Campaign", 1, 50, 2)
    pdays = st.number_input("Days Since Previous Contact (-1 if none)", -1, 500, -1)
    previous = st.number_input("Number of Previous Contacts", 0, 50, 0)
    poutcome = st.selectbox(
        "Outcome of Previous Campaign", ["failure", "success", "other", "unknown"]
    )

# Prediction button
if st.button("🧠 Predict"):
    input_payload = [
        {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
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

# Drift reports
st.subheader("📊 Drift Reports")
drift_type = st.radio(
    "Select drift report to view:", ("Data Drift", "Prediction Drift")
)
drift_filename = "data_drift.html" if drift_type == "Data Drift" else "pred_drift.html"
drift_report_path = Path("../../../.drift_reports") / drift_filename

if drift_report_path.exists():
    with st.expander(f"📉 {drift_type} Report"):
        st.components.v1.html(drift_report_path.read_text(), height=800, scrolling=True)
else:
    st.info(f"{drift_type} report not yet available.")
