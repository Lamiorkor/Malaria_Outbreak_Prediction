FROM python:3.11-slim

WORKDIR /app

COPY dockerisation_and_deployment/webservices/main.py .
COPY dockerisation_and_deployment/webservices/predict.py .
COPY dockerisation_and_deployment/webservices/__init__.py .
COPY models /app/models

COPY requirements.txt .
RUN pip install -r requirements.txt

# ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001
# ENV MLFLOW_REGISTRY_URI=http://host.docker.internal:5001
ENV BASE_DIR=/app
ENV MODEL_DIR=/app/models

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]