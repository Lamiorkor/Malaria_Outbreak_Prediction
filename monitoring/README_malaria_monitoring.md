# Malaria Monitoring Pipeline

## Folder structure

```text
05_monitoring/
├── data/
│   ├── reference.csv
│   └── current_batches/
├── scripts/
│   ├── prepare_reference.py
│   ├── generate_batch.py
│   └── calculate_metrics.py
├── docker-compose.yml
└── README.md
```

## What each script does

- `prepare_reference.py`
  - builds a baseline/reference dataset from historical malaria rows
  - generates model predictions using your existing `PredictionPipeline`
  - creates a proxy `true_label` for backtesting:
    - outbreak next year = 1 if next year's incidence is at or above the chosen quantile threshold

- `generate_batch.py`
  - samples a new batch from later years
  - optionally injects synthetic drift (`none`, `climate`, `economic`, `mixed`)
  - runs predictions and saves a batch CSV

- `calculate_metrics.py`
  - compares the latest batch to `reference.csv`
  - computes drift, performance, prediction shares, and average latency
  - stores one row per batch in PostgreSQL table `monitoring_metrics`

## Recommended live demo flow

### 1) Prepare the reference once
```bash
python scripts/prepare_reference.py --raw-path malaria_final_dataset.csv
```

### 2) Generate a new batch live
```bash
python scripts/generate_batch.py --batch-size 20 --drift-mode mixed
```

### 3) Calculate and store the monitoring metrics live
```bash
python scripts/calculate_metrics.py
```

### 4) Refresh Adminer / Grafana
- Adminer query:
```sql
SELECT * FROM monitoring_metrics ORDER BY timestamp DESC;
```

## Suggested Grafana panels

### Total processed batches
```sql
SELECT COUNT(*) AS total_batches
FROM monitoring_metrics;
```

### Accuracy over time
```sql
SELECT timestamp AS "time", accuracy
FROM monitoring_metrics
ORDER BY timestamp;
```

### Share of drifted features over time
```sql
SELECT timestamp AS "time", share_drifted_features
FROM monitoring_metrics
ORDER BY timestamp;
```

### High-risk prediction share over time
```sql
SELECT timestamp AS "time", pred_high_risk_share
FROM monitoring_metrics
ORDER BY timestamp;
```

### Average latency over time
```sql
SELECT timestamp AS "time", avg_latency_ms
FROM monitoring_metrics
ORDER BY timestamp;
```

