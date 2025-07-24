# Naive Bayes Customer Purchase Prediction

A microservices-based machine learning application that predicts whether customers will make purchases based on their demographic and behavioral data. The system uses a custom implementation of the Naive Bayes algorithm.

## Project Features

- **Custom Naive Bayes Implementation** - Built from scratch without external ML libraries
- **Microservices Architecture** - Separate services for training and prediction
- **Data Processing Pipeline** - Automated data cleaning and preprocessing
- **Model Evaluation** - Built-in accuracy metrics and confusion matrix
- **RESTful APIs** - Easy integration with web applications
- **Docker Support** - Containerized deployment

## Data Format

The model expects customer data with these fields:
- `Age`: Customer age (0-120)
- `Gender`: Customer gender 
- `IncomeLevel`: Income category
- `MaritalStatus`: Marital status
- `PreviousPurchases`: Number of past purchases
- `VisitedWebsite`: Whether customer visited website
- `WillBuy`: Target variable (Yes/No)

## Services

- **Model API** (port 8000) - Handles data loading, model training, and model serving
- **Predict API** (port 8001) - Makes predictions using the trained model

## Quick Start with Docker


### Manual Docker
```bash
# Create network
docker network create class-app

# Build and run model service
docker build -f Dockerfile.model -t model-service .
docker run -d --name model-api --network class-app -p 8000:8000 model-service

# Build and run predict service
docker build -f Dockerfile.predict -t predict-service .
docker run -d --name predict-api --network class-app -p 8001:8001 predict-service
```

## API Endpoints

### Model Service (port 8000)

**Get Model Accuracy:**
```bash
curl http://localhost:8000/precision
```
Returns: `{"model precision": 85.5}`

**Get Confusion Matrix:**
```bash
curl http://localhost:8000/matrix
```
Returns detailed metrics including accuracy, precision, recall, and F1-score.

**Download Model:**
```bash
curl http://localhost:8000/model --output model.pkl
```

### Prediction Service (port 8001)

**Make Prediction:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Gender": "Male",
    "IncomeLevel": "High",
    "MaritalStatus": "Married",
    "PreviousPurchases": 5,
    "VisitedWebsite": "Yes"
  }'
```

## Stop Services

```bash
# Manual
docker stop model-api predict-api
docker rm model-api predict-api
```