# 🧠 Breast Cancer Prediction and Monitoring System

This project provides a complete pipeline for predicting breast cancer using a trained machine learning model, served via a FastAPI backend and a Streamlit web frontend. It also includes monitoring capabilities for **data drift** and **prediction drift**, using the [Evidently AI](https://www.evidentlyai.com/) library.

---

## 📦 Project Structure

```

.
├── config/                   # YAML configuration files for different models
├── notebooks/                # Original exploratory notebook
├── src/
│   ├── api/                  # FastAPI application
│   ├── apps/                 # Entrypoints (e.g. CLI, Streamlit UI)
│   ├── artifacts/            # Saving model + scaler + training data
│   ├── data/                 # Data loading logic
│   ├── evaluation/           # Model evaluation and metrics
│   ├── features/             # Feature engineering and preprocessing
│   ├── models/               # Model creation and training
│   ├── monitoring/           # Drift detection logic
│   ├── pipeline/             # Orchestration of the training/inference flows
│   ├── schemas/              # Request and response schemas
│   └── utils/                # Logger, config loader, and utilities
├── tests/                    # Unit tests
├── Dockerfile.api            # Dockerfile for FastAPI service
├── Dockerfile.ui             # Dockerfile for Streamlit UI
├── docker-compose.yml        # Compose setup to run both services
├── pyproject.toml            # Python project metadata and dependencies
└── README.md                 # This file

````

---

## 🚀 Features

- ✅ FastAPI backend for real-time predictions
- ✅ Streamlit frontend with interactive UI
- ✅ Evidently integration for:
  - 📈 **Data drift** monitoring
  - 📉 **Prediction drift** monitoring
- ✅ Persistent drift logging
- ✅ Scalable architecture using Docker and Docker Compose

---

## ⚙️ Setup

### Requirements

- Docker
- Docker Compose

### 1. Build and run services

```bash
docker-compose up --build
````

---

## 🌐 Services

### 🔹 FastAPI (Backend)

* **URL:** `http://localhost:8000`
* **Endpoint:** `POST /predict`
* Accepts JSON payload of input features and returns predictions + probability.

### 🔸 Streamlit (Frontend)

* **URL:** `http://localhost:8501`
* Interactive form to submit patient data and view:

  * Model predictions
  * Data and prediction drift reports (via `evidently`)

---

## 📊 Monitoring

Drift reports are saved to the `.drift_reports/` directory and rendered in Streamlit:

* `data_drift.html`
* `pred_drift.html`

The drift is calculated automatically as new predictions are made, using buffered batches of data.

---

## 🐳 Docker Notes

### Volumes

The following volumes are mounted to persist and share data:

```yaml
volumes:
  - ./config:/app/config
  - ./.drift_reports:/app/drift_reports
  - ./.artifacts:/app/models
```

Ensure the folders `.drift_reports/` and `.artifacts/` exist before running the app.

---

## 📬 Example Request

```json
POST /predict
Content-Type: application/json

[
  {
    "mean_radius": 14.5,
    "mean_texture": 20.5,
    "mean_perimeter": 96.2,
    ...
  }
]
```

Response:

```json
[
  {
    "prediction": 1,
    "prediction_prob": 0.9923,
    "label": "benign"
  }
]
```

---

## 🧪 Development

You can run the app manually with:

```bash
uv run uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000
streamlit run src/apps/app_streamlit.py
```

---

## 📁 Config

Edit the configuration in `config/config.yaml` to tune:

* Artifacts paths
* Drift reports paths
* Drift buffer sizes
