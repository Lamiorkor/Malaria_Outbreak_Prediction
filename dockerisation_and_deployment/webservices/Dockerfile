FROM python:3.9-slim

WORKDIR /app

COPY dockerisation_and_deployment/webservices/main.py .
COPY dockerisation_and_deployment/webservices/predict.py .
COPY dockerisation_and_deployment/webservices/__init__.py .
COPY models /app/models

RUN pip install --no-cache-dir mlflow joblib numpy pandas scikit-learn fastapi uvicorn

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001
ENV MLFLOW_REGISTRY_URI=http://host.docker.internal:5001
ENV BASE_DIR=/app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]