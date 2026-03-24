# Malaria Outbreak Prediction System
A paired Machine Learning Ops project for the FS26 Artificial Intelligence class at HSLU


## Overview

This project presents an end-to-end MLOps pipeline for predicting malaria outbreak risk using climate, demographic, and historical malaria data.

The system combines:
* Machine Learning (Logistic Regression)
* Feature Engineering (temporal + climate features)
* Experiment Tracking (MLflow)
* API Deployment (FastAPI)
* Containerization (Docker)
* Batch Processing (Scheduled training & prediction)

---

## Objective

To predict the probability of a malaria outbreak for a given country-year and trigger an alert when risk exceeds a defined threshold.

---

## Key Features

* Advanced feature engineering:

  * Lag features (Malaria_Lag1–3)
  * Rolling averages (3-year windows)
  * Climate interactions
  * Log transformations

* Class imbalance handling:

  * Logistic Regression with `class_weight="balanced"`

* Experiment tracking:

  * MLflow used to log parameters, metrics, and models

* Reproducible pipelines:

  * Modular training and prediction pipelines

* API deployment:

  * FastAPI for real-time predictions

* Containerized service:

  * Docker for portability and deployment

* Batch workflow:

  * Scheduled retraining + prediction simulation

---


## Model Details

**Final Model:** Logistic Regression

```python
LogisticRegression(
    C=1.0,
    class_weight="balanced",
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
```

### Performance

| Metric    | Validation | Test  |
| --------- | ---------- | ----- |
| AUC-ROC   | 0.993      | 0.998 |
| F1 Score  | 0.980      | 0.935 |
| Recall    | 1.000      | 1.000 |
| Precision | 0.960      | 0.878 |

---

## Prediction Logic

The model outputs a probability of outbreak, which is converted into an alert:

```text
alert = probability ≥ threshold
```

* Threshold = **0.5**
  
* Risk Levels:
  * LOW (< 0.40)
  * MEDIUM (0.40–0.69)
  * HIGH (≥ 0.70)

---

## MLOps Workflow

```text
Raw Data
   ↓
Feature Engineering
   ↓
Feature Store (CSV + scaler.pkl)
   ↓
MLflow Experiment Tracking
   ↓
Training Pipeline
   ↓
Model Artifacts (model + scaler + metadata)
   ↓
FastAPI Service
   ↓
Docker Container
   ↓
Batch Pipeline (Scheduled Training + Prediction)
```

---

## How to Run

### 1. Train the model

```bash
python3 training_pipeline/train_pipeline.py
```

---

### 2. Start FastAPI

```bash
cd dockerisation_and_deployment/webservices
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

### 3. Run batch workflow

```bash
python3 dockerisation_and_deployment/batch/train_predict_scheduled.py
```

---

### 4. Run with Docker

```bash
docker build -f dockerisation_and_deployment/webservices/Dockerfile -t malaria-api .
docker run -p 8000:8000 malaria-api
```

---

## Output Example

```json
{
  "country": "Tanzania",
  "year": 2023,
  "outbreak_probability": 0.1500,
  "outbreak_alert": false,
  "risk_level": "LOW"
}
```

---

## Key MLOps Principles Applied

* Reproducibility (pipelines + feature store)
* Separation of concerns (training vs inference)
* Model versioning (MLflow)
* Deployment abstraction (API + Docker)
* Automation (batch workflow)
* Consistency (shared scaler across pipelines)

---

## Future Improvements

* Replace file-based MLflow backend with database
* Add model monitoring (data drift, performance tracking)
* Implement CI/CD pipeline
* Integrate real-time data ingestion
* Add multiple model comparison (XGBoost, LSTM)

---

## Authors

**Likhita Kolli**
MSc IT, Digitalisation & Sustainability
Lucerne University of Applied Sciences and Arts (HSLU)

**Naa Lamiorkor Boye**
MSc IT, Digitalisation & Sustainability
Lucerne University of Applied Sciences and Arts (HSLU)

**Thank you :)**
