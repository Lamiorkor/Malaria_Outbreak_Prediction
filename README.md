# 🌍 Climate-Driven Malaria Outbreak Prediction System

An end-to-end MLOps project developed for the Artificial Intelligence (ARTIFIN) course at the Lucerne University of Applied Sciences and Arts (HSLU).

This project builds a production-oriented machine learning system capable of predicting malaria outbreak risk using climate, demographic, and historical malaria incidence data. Beyond model training, the project implements the complete machine learning lifecycle, including experiment tracking, deployment, monitoring, automation, testing, containerization, and CI/CD.

---

## 📖 Overview

Climate change is reshaping disease transmission patterns across the world. Rising temperatures, changing rainfall patterns, and shifting environmental conditions influence the spread of vector-borne diseases such as malaria.

Current public health responses are often reactive, responding only after outbreaks have begun.

This project addresses that challenge by developing an AI-powered early warning system that predicts the probability of a malaria outbreak for a given country-year using climate, environmental, and socioeconomic indicators.

The system combines machine learning with modern MLOps practices to create a reproducible, deployable, monitorable, and maintainable solution.

---

# 🎯 Project Objective

Predict the probability of a malaria outbreak occurring in the next period using:

- Historical malaria incidence
- Climate variables
- Population data
- Economic indicators

The goal is to support proactive decision-making by enabling:

- Early outbreak detection
- Resource planning
- Public health preparedness
- Climate-health risk assessment

---

# 🏥 Real-World Relevance

Climate change is expanding mosquito habitats and altering disease transmission patterns worldwide.

Potential users of systems like this include:

- Ministries of Health
- World Health Organization (WHO)
- NGOs
- Public health agencies
- Climate adaptation initiatives
- Disease surveillance programs

The project sits at the intersection of:

- Artificial Intelligence
- Public Health
- Epidemiology
- Climate Science
- Data Engineering
- MLOps

---

# 🛠 Technology Stack

## Machine Learning

- Logistic Regression
- Scikit-Learn
- Pandas
- NumPy

## Experiment Tracking

- MLflow

## API Deployment

- FastAPI
- Uvicorn

## Containerization

- Docker

## Container Orchestration

- Docker Compose

## Workflow Automation

- Prefect

## Database

- PostgreSQL

## Monitoring

- Evidently AI
- Grafana

## Testing

- Pytest

## CI/CD

- GitHub Actions
- GitHub Container Registry (GHCR)

---

# 🧠 Machine Learning Pipeline

## Data Sources

The model uses:

- Historical malaria incidence data
- Climate indicators
- Population density data
- GDP per capita data

These datasets are merged and transformed into a machine-learning-ready feature store.

---

## Feature Engineering

The final model uses a combination of climate, temporal, and socioeconomic features.

### Base Features

- Malaria_Incidence
- Precipitation_mm
- Pop_Density
- GDP_per_Capita
- Temp_Annual_Mean_C
- Temp_GrowingSeason_Mean_C

### Lag Features

- Malaria_Lag1
- Malaria_Lag2
- Malaria_Lag3
- Temp_Lag1
- Precip_Lag1
- GDP_Lag1

### Rolling Features

- Malaria_Roll3
- Temp_Roll3
- Precip_Roll3

### Derived Features

- Malaria_YoY_Change
- Temp_Precip_Interaction
- Temp_Squared
- Climate_Risk_Index
- Vulnerability_Index
- Log_GDP
- Log_Malaria_Incidence

---

# 🤖 Model Details

## Final Model

```python
LogisticRegression(
    C=1.0,
    class_weight="balanced",
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
```

## Model Performance

| Metric | Validation | Test |
|----------|----------|----------|
| AUC-ROC | 0.993 | 0.998 |
| F1 Score | 0.980 | 0.935 |
| Recall | 1.000 | 1.000 |
| Precision | 0.960 | 0.878 |

---

# 📤 Prediction Output

The system predicts:

```json
{
  "country": "Ghana",
  "year": 2023,
  "outbreak_probability": 0.42,
  "outbreak_alert": false,
  "risk_level": "LOW"
}
```

### Alert Threshold

```text
outbreak_alert = probability >= 0.5
```

### Risk Categories

| Probability | Risk Level |
|------------|------------|
| < 0.40 | LOW |
| 0.40 - 0.69 | MEDIUM |
| ≥ 0.70 | HIGH |

---

# 🔬 Experiment Tracking

MLflow is used to track:

- Hyperparameters
- Training runs
- Metrics
- Model artifacts
- Model metadata

Artifacts stored include:

- Logistic Regression model
- Feature scaler
- Model metadata

This ensures reproducibility and traceability across experiments.

---

# 🚀 Deployment

The trained model is served through a FastAPI application.

## Available Endpoints

### Health Check

```http
GET /
```

### Model Information

```http
GET /model/info
```

### Prediction Endpoint

```http
POST /predict
```

---

# 🐳 Containerization

The application is containerized using Docker.

Benefits include:

- Reproducibility
- Environment consistency
- Easy deployment
- Simplified dependency management

Build image:

```bash
docker build -t malaria-api .
```

Run container:

```bash
docker run -p 8000:8000 malaria-api
```

---

# 🏗 Docker Compose Architecture

The complete local stack is orchestrated using Docker Compose.

Services:

### FastAPI

Serves prediction requests.

### MLflow

Tracks experiments and model artifacts.

### PostgreSQL

Stores prediction logs and monitoring information.

Architecture:

```text
                 ┌─────────────┐
                 │   MLflow    │
                 └──────▲──────┘
                        │
                        │
┌─────────────┐  ┌──────┴──────┐  ┌─────────────┐
│   Client    │─▶│ malaria-api │─▶│ PostgreSQL  │
└─────────────┘  └─────────────┘  └─────────────┘
                    :8000             :5433
```

Start the stack:

```bash
docker compose up --build
```

---

# 🗄 Prediction Logging

Every prediction request is logged to PostgreSQL.

Stored information includes:

- Country
- Year
- Input features
- Lag values
- Outbreak probability
- Prediction result
- Model version
- Inference latency
- Timestamp

This creates an auditable prediction history and supports monitoring workflows.

---

# 📊 Monitoring & Drift Detection

A dedicated monitoring pipeline evaluates model health after deployment.

## Monitoring Features

- Data drift detection
- Feature drift monitoring
- Batch-based evaluation
- Statistical testing
- Historical monitoring logs

## Drift Detection

The project uses:

- Kolmogorov-Smirnov (KS) tests
- Reference vs Current dataset comparison

to identify significant distribution shifts.

## Monitoring Outputs

- Drift metrics
- Feature-level drift reports
- Monitoring dashboards
- Alert generation

---

# 📈 Grafana Dashboards

Grafana is used to visualize:

- Prediction volumes
- Drift metrics
- Model health indicators
- Monitoring statistics

This provides operational visibility into the deployed system.

---

# ⏰ Workflow Automation

The project implements automated scheduling using Prefect.

Implemented workflows include:

- Scheduled model execution
- Automated prediction jobs
- Batch processing workflows

Although deployment scheduling was demonstrated during the project, the implementation showcases how machine learning workflows can be orchestrated in production environments.

---

# 🧪 Testing Strategy

The project includes automated testing covering multiple levels of the system.

## Unit Tests

Validate:

- Prediction logic
- Input validation
- Feature count verification
- Error handling

## API Contract Tests

Validate:

- Request schema
- Field types
- Value bounds
- Required fields

## Smoke Tests

Validate:

- End-to-end prediction pipeline
- Feature engineering
- Model inference
- Output structure

---

# 🔄 CI/CD Pipeline

GitHub Actions automates testing and deployment.

## Continuous Integration (CI)

On every push:

- Syntax validation
- Dependency installation
- Unit tests
- API contract tests
- Smoke tests

## Continuous Deployment (CD)

After successful CI:

- Docker image build
- Image tagging
- Image publishing to GitHub Container Registry
- Container smoke testing

Pipeline flow:

```text
Push
 ↓
Run Tests
 ↓
Build Docker Image
 ↓
Publish to GHCR
 ↓
Smoke Test Container
```

---

# 📂 Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── dockerisation_and_deployment/
│   ├── batch/
│   └── webservices/
│
├── model_tracking/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
│
├── monitoring/
│
├── tests/
│
├── training_pipeline/
│
├── docker-compose.yml
├── init.sql
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

---

# 🔁 MLOps Lifecycle Coverage

Unlike traditional machine learning projects that stop after model training, this project implements the full machine learning lifecycle.

✅ Data Preparation

✅ Feature Engineering

✅ Model Training

✅ Experiment Tracking

✅ API Deployment

✅ Containerization

✅ Workflow Automation

✅ Monitoring

✅ Drift Detection

✅ Automated Testing

✅ CI/CD

This transforms the project from a machine learning model into a production-oriented machine learning system.

---

# 🚀 Future Improvements

Potential future enhancements include:

- MLflow Model Registry integration
- Automatic retraining triggers based on drift thresholds
- Cloud deployment (AWS, Azure, GCP)
- Real-time streaming inference
- Advanced forecasting models (XGBoost, LSTM, TFT)
- SHAP explainability dashboards
- Multi-disease prediction support

---

# 👥 Authors

### Likhita Kolli

MSc IT, Digitalisation & Sustainability  
Lucerne University of Applied Sciences and Arts (HSLU)

### Naa Lamiorkor Boye

MSc IT, Digitalisation & Sustainability  
Lucerne University of Applied Sciences and Arts (HSLU)

---

## 🙏 Acknowledgements

Developed as part of the Artificial Intelligence (ARTIFIN) module at the Lucerne University of Applied Sciences and Arts (HSLU), with a focus on applying modern MLOps principles to a real-world climate and public health challenge.

**Thank you for visiting our project!**
