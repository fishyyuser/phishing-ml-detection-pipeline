# End-to-End Phishing URL Detection — Modular, Production-Ready ML Pipeline & Deployment System

This project implements a fully modular, end-to-end classical ML system for phishing URL detection — from raw CSV ingestion to a deployed inference API — built using a production-first ML engineering architecture.

The design emphasizes:
- Component-based pipeline structure
- Clear separation between training and inference
- Clean artifact + config management
- Schema-driven data validation
- Production-style logging & exception handling
- FastAPI for serving
- Docker for deployment

This mirrors real-world ML engineering standards: modular, testable, maintainable, and deployment-ready.



## Problem Statement

The system classifies URLs as **phishing** or **legitimate** using a classical ML model.

Raw labeled URLs pass through a multi-stage training pipeline, and the deployed FastAPI inference service returns:
- predicted class
- confidence score

Consistent preprocessing ensures training-serving parity.



## Model Summary

**Final Model:** Random Forest Classifier

**Test Metrics:**
- F1 Score: 0.9741
- Precision: 0.9737
- Recall: 0.9745

All metrics and artifacts are tracked using **MLflow**.



## API Inputs

The deployed API accepts:
- **URL string**

The service returns:
- predicted label (`phishing` / `legitimate`)
- confidence score

A simple web UI (Jinja templates) supports single URL input and batch CSV prediction.



# Architecture Overview
```bash
Training Pipeline (main.py)
↓
Saved Artifacts (model, transformer, metrics)
↓
Inference Pipeline (loads artifacts only)
↓
FastAPI Server (runtime prediction)
↓
Docker Deployment
```


### Key Engineering Decisions
- Strict separation of training and inference
- Component-based modular pipeline
- Schema-based validation (YAML)
- Logged experiments via MLflow
- Production-grade logging + error handling
- Lightweight Dockerized inference environment



# Training Pipeline Components

1. **Data Ingestion**  
   Reads raw CSV, splits into train/test.

2. **Data Validation**  
   Validates structure using `schema.yaml`.

3. **Data Transformation**  
   Feature engineering + preprocessing for classical ML.

4. **Model Training**  
   Trains multiple classical ML models via GridSearchCV.

5. **Model Evaluation**  
   Computes F1/Precision/Recall, logs results to MLflow.

6. **Model Pushing**  
   Saves model + transformer artifacts using `dill`.

Artifacts are stored in a versioned structure inside `Artifacts/`.



# Deployment

A **FastAPI** app (`app.py`) serves real-time predictions.

### Run Locally
```bash
python app.py
```
Open:
```bash
http://localhost:8000
```
Supports:
- UI-based predictions
- JSON API predictions
- CSV batch predictions



## Docker Deployment

### Build Image
```bash
docker build -t phishing-detector .
```

### Run Container
```bash
docker run -p 8000:8000 \
  -e MONGO_DB_URL="$MONGO_DB_URL" \
  -v /path/to/artifacts:/app/Artifacts \
  phishing-detector
```

Open:
```bash
http://localhost:8000
```

## EC2 Deployment Example

### SSH into instance
```bash
ssh -i <key.pem> ubuntu@<public-ip>
```

### Run the Docker container
```bash
docker run -d -p 8000:8000 \
  -e MONGO_DB_URL="mongodb://..." \
  -v /home/ubuntu/artifacts:/app/Artifacts \
  phishing-detector
```
Visit:
```bash
http://<public-ip>:8000
```



## Tech Stack

### Core
- Python
- Scikit-Learn
- Pandas / NumPy

### Pipeline
- Modular component-based architecture
- YAML schema validation
- Dill-based artifact serialization

### Experiment Tracking
- MLflow

### Serving
- FastAPI
- Uvicorn
- Jinja2 templates

### Deployment
- Docker
- Optional MongoDB logging



## License

MIT
