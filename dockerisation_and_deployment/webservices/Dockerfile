FROM python:3.11-slim

WORKDIR /app

# Copy full project structure 
COPY dockerisation_and_deployment /app/dockerisation_and_deployment
COPY models /app/models

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Environment variables
ENV BASE_DIR=/app
ENV MODEL_DIR=/app/models

EXPOSE 8000

# Run using full module path
CMD ["uvicorn", "dockerisation_and_deployment.webservices.main:app", "--host", "0.0.0.0", "--port", "8000"]