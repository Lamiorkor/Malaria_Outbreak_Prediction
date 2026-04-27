# Malaria Monitoring Pipeline

## 📌 Overview

This module implements the monitoring component of the **Climate-Driven Disease Outbreak Prediction System**.

The goal is to track model performance over time and detect potential degradation due to:

* Data drift (changes in input distribution)
* Prediction drift
* Model performance decay

This ensures the system remains reliable in a real-world deployment setting.

---

## 🏗️ Monitoring Architecture

The monitoring pipeline consists of:

1. **Reference Dataset**

   * Historical malaria data
   * Used as baseline (ground truth approximation)

2. **Current Batch Data**

   * Simulated incoming data
   * Generated using `generate_batch.py`

3. **Metrics Calculation**

   * Performance metrics (accuracy, precision, recall)
   * Data drift metrics

4. **Storage**

   * Metrics stored in PostgreSQL

5. **Visualization**

   * Grafana dashboards for monitoring trends

---

## 📂 Project Structure

```
monitoring/
│
├── docker-compose.yaml        # Monitoring stack (Postgres, Grafana)
├── README_malaria_monitoring.md
├── ML_MonitoringPipeline.jpeg
├── ML_Monitoring_Pipeline_Diagram.png
│
└── scripts/
    ├── prepare_reference.py   # Create baseline dataset
    ├── generate_batch.py      # Simulate incoming data
    └── calculate_metrics.py   # Compute monitoring metrics
```

---

## ⚙️ Setup Instructions

### 1️⃣ Start Monitoring Stack

```bash
cd monitoring
docker-compose up --build
```

This will start:

* PostgreSQL (metrics storage)
* Grafana (visualization)

---

### 2️⃣ Prepare Reference Dataset

```bash
python scripts/prepare_reference.py
```

This creates the baseline dataset used for comparison.

---

### 3️⃣ Generate Batch Data

```bash
python scripts/generate_batch.py --drift-mode none --batch-size 30
```

You can simulate different conditions:

* `none` → normal data
* `drift` → shifted distribution

---

### 4️⃣ Calculate Metrics

```bash
python scripts/calculate_metrics.py
```

This computes:

* Model performance metrics
* Drift indicators

---

## 📊 Metrics Tracked

### Model Performance

* Accuracy
* Precision
* Recall

### Data Monitoring

* Feature distribution changes
* Drift detection indicators

---

## 📈 Visualization (Grafana)

Access Grafana at:

```
http://localhost:3000
```

Default credentials:

```
admin / admin
```

Dashboards show:

* Performance trends over time
* Drift signals
* Batch-level monitoring insights

---

## 🔁 Monitoring Workflow

1. Generate new batch data
2. Run predictions
3. Calculate metrics
4. Store results in database
5. Visualize in Grafana
6. Trigger retraining if degradation detected

---

## 🚀 Future Improvements

* Add automated drift thresholds
* Integrate alerting (email/Slack)
* Use Evidently AI for advanced monitoring
* Automate pipeline with Prefect

---

## 🎯 Key Takeaways

This monitoring system demonstrates:

* End-to-end MLOps lifecycle
* Production-level model observability
* Practical handling of data drift and performance decay
