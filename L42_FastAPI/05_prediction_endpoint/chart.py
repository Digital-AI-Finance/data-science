"""Prediction Endpoint - Complete ML API"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prediction Endpoint: Complete ML API', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Complete prediction API
ax1 = axes[0, 0]
ax1.axis('off')

complete_api = '''
COMPLETE ML PREDICTION API

# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np


# Load model at startup
model = joblib.load('models/stock_classifier.joblib')
features = joblib.load('models/features.joblib')


# Schemas
class StockInput(BaseModel):
    price: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    momentum: float = Field(default=0.0)
    volatility: float = Field(default=0.0)

    class Config:
        schema_extra = {
            "example": {
                "price": 150.5,
                "volume": 1000000,
                "momentum": 0.02,
                "volatility": 0.15
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict


# Create app
app = FastAPI(
    title="Stock Prediction API",
    description="ML model for stock direction prediction",
    version="1.0.0"
)


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StockInput):
    \"\"\"Make a prediction for stock direction.\"\"\"

    # Prepare features
    X = np.array([[
        data.price,
        data.volume,
        data.momentum,
        data.volatility
    ]])

    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return PredictionResponse(
        prediction="buy" if pred == 1 else "sell",
        confidence=float(max(proba)),
        probabilities={"sell": proba[0], "buy": proba[1]}
    )
'''

ax1.text(0.02, 0.98, complete_api, transform=ax1.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Complete Prediction API', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Additional endpoints
ax2 = axes[0, 1]
ax2.axis('off')

additional = '''
ADDITIONAL USEFUL ENDPOINTS

# Health check
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


# Model information
@app.get("/model/info")
def model_info():
    return {
        "name": "stock_classifier",
        "version": "2.0.0",
        "algorithm": "RandomForest",
        "features": features,
        "classes": ["sell", "buy"]
    }


# Batch predictions
@app.post("/predict/batch")
def predict_batch(data: List[StockInput]):
    \"\"\"Predict for multiple stocks at once.\"\"\"
    results = []
    for item in data:
        result = predict(item)
        results.append(result)
    return {"predictions": results, "count": len(results)}


# Feature validation
@app.get("/model/features")
def get_features():
    return {
        "required": features,
        "schema": {
            "price": {"type": "float", "min": 0},
            "volume": {"type": "int", "min": 0},
            "momentum": {"type": "float", "default": 0},
            "volatility": {"type": "float", "default": 0}
        }
    }


# Async prediction (for long-running tasks)
from fastapi import BackgroundTasks

@app.post("/predict/async")
async def predict_async(
    data: StockInput,
    background_tasks: BackgroundTasks
):
    task_id = create_task_id()
    background_tasks.add_task(run_prediction, task_id, data)
    return {"task_id": task_id, "status": "processing"}
'''

ax2.text(0.02, 0.98, additional, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Additional Endpoints', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Testing the API
ax3 = axes[1, 0]
ax3.axis('off')

testing = '''
TESTING THE API

# Using curl:
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
         "price": 150.5,
         "volume": 1000000,
         "momentum": 0.02,
         "volatility": 0.15
     }'

# Response:
{
    "prediction": "buy",
    "confidence": 0.85,
    "probabilities": {"sell": 0.15, "buy": 0.85}
}


# Using Python requests:
import requests

url = "http://localhost:8000/predict"
data = {
    "price": 150.5,
    "volume": 1000000,
    "momentum": 0.02,
    "volatility": 0.15
}

response = requests.post(url, json=data)
print(response.json())


# Batch request:
batch_data = [
    {"price": 150.5, "volume": 1000000},
    {"price": 145.0, "volume": 2000000},
    {"price": 160.0, "volume": 500000}
]

response = requests.post(
    "http://localhost:8000/predict/batch",
    json=batch_data
)
print(response.json())


# Test with pytest:
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={
        "price": 150.5,
        "volume": 1000000
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
'''

ax3.text(0.02, 0.98, testing, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Testing the API', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: API flow visualization
ax4 = axes[1, 1]
ax4.axis('off')

# Draw request-response flow
steps = [
    (0.1, 'Client\nRequest'),
    (0.3, 'Pydantic\nValidation'),
    (0.5, 'Endpoint\nFunction'),
    (0.7, 'Model\nPredict'),
    (0.9, 'JSON\nResponse')
]

y = 0.7
for x, label in steps:
    color = MLBLUE if 'Client' in label or 'Response' in label else MLGREEN if 'Model' in label else MLORANGE
    ax4.add_patch(plt.Rectangle((x-0.08, y-0.1), 0.16, 0.2, facecolor=color, alpha=0.3))
    ax4.text(x, y, label, fontsize=9, ha='center', va='center')

# Arrows
for i in range(len(steps)-1):
    ax4.annotate('', xy=(steps[i+1][0]-0.08, y), xytext=(steps[i][0]+0.08, y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

# Labels
ax4.text(0.3, y+0.2, 'Auto validation!', fontsize=8, ha='center', color=MLORANGE)
ax4.text(0.7, y+0.2, 'ML prediction', fontsize=8, ha='center', color=MLGREEN)

# Show example request/response
request_example = '''
Request:
{"price": 150.5, "volume": 1000000}
'''
response_example = '''
Response:
{"prediction": "buy", "confidence": 0.85}
'''

ax4.text(0.1, 0.35, request_example, fontsize=8, fontfamily='monospace',
         bbox=dict(facecolor=MLBLUE, alpha=0.2))
ax4.text(0.65, 0.35, response_example, fontsize=8, fontfamily='monospace',
         bbox=dict(facecolor=MLGREEN, alpha=0.2))

ax4.set_xlim(0, 1)
ax4.set_ylim(0.2, 1)
ax4.set_title('Request-Response Flow', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
