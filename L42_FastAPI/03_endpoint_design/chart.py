"""Endpoint Design - REST API Best Practices"""
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
fig.suptitle('Endpoint Design: REST API Best Practices', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: URL naming conventions
ax1 = axes[0, 0]
ax1.axis('off')

naming = '''
URL NAMING CONVENTIONS

GOOD PRACTICES:
---------------
- Use nouns, not verbs (resources)
- Plural names for collections
- Lowercase with hyphens
- Hierarchical structure


GOOD EXAMPLES:
--------------
GET  /api/v1/stocks           # List all stocks
GET  /api/v1/stocks/AAPL      # Get specific stock
POST /api/v1/predictions      # Create prediction
GET  /api/v1/models           # List models
GET  /api/v1/models/v2/info   # Get model info


BAD EXAMPLES:
-------------
GET /getStocks          # Verb in URL
GET /api/Stock          # Singular, uppercase
POST /api/makePrediction # Verb, camelCase
GET /api/v1/get-all-stocks # Redundant


FOR ML MODELS:
--------------
POST /api/v1/predict           # Make prediction
POST /api/v1/predict/batch     # Batch predictions
GET  /api/v1/model/info        # Model metadata
GET  /api/v1/model/features    # Required features
GET  /api/v1/model/health      # Health check


VERSIONING:
-----------
Include version in URL!

/api/v1/predict  <- Version 1
/api/v2/predict  <- Version 2 (breaking changes)

This allows backward compatibility.
'''

ax1.text(0.02, 0.98, naming, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('URL Naming Conventions', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: ML API structure
ax2 = axes[0, 1]
ax2.axis('off')

structure = '''
ML API ENDPOINT STRUCTURE

RECOMMENDED ENDPOINTS FOR ML:
-----------------------------
from fastapi import FastAPI

app = FastAPI(
    title="Stock Prediction API",
    version="1.0.0"
)


# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}


# Model info
@app.get("/api/v1/model/info")
def model_info():
    return {
        "name": "stock_classifier",
        "version": "2.0.0",
        "features": ["price", "volume", "momentum"],
        "last_trained": "2024-01-15"
    }


# Single prediction
@app.post("/api/v1/predict")
def predict(data: PredictionInput):
    prediction = model.predict(data)
    return {"prediction": prediction}


# Batch predictions
@app.post("/api/v1/predict/batch")
def predict_batch(data: List[PredictionInput]):
    predictions = [model.predict(d) for d in data]
    return {"predictions": predictions}


# Feature requirements
@app.get("/api/v1/model/features")
def get_features():
    return {
        "required": ["price", "volume"],
        "optional": ["momentum"],
        "types": {
            "price": "float",
            "volume": "int"
        }
    }
'''

ax2.text(0.02, 0.98, structure, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('ML API Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Status codes
ax3 = axes[1, 0]

# Create status codes visualization
status_codes = [
    ('200', 'OK', 'Success', MLGREEN),
    ('201', 'Created', 'Resource created', MLGREEN),
    ('400', 'Bad Request', 'Invalid input', MLORANGE),
    ('404', 'Not Found', 'Resource not found', MLORANGE),
    ('422', 'Validation Error', 'Invalid data type', MLORANGE),
    ('500', 'Server Error', 'Model crashed', MLRED)
]

y_positions = np.arange(len(status_codes))
codes = [s[0] for s in status_codes]
descriptions = [f"{s[1]}: {s[2]}" for s in status_codes]
colors = [s[3] for s in status_codes]

bars = ax3.barh(y_positions, [1]*len(status_codes), color=colors, alpha=0.6, edgecolor='black')

ax3.set_yticks(y_positions)
ax3.set_yticklabels([f"{s[0]} - {s[1]}" for s in status_codes], fontsize=10)
ax3.set_xlim(0, 2)
ax3.set_xticks([])

# Add descriptions
for i, desc in enumerate([s[2] for s in status_codes]):
    ax3.text(1.1, i, desc, fontsize=9, va='center')

ax3.invert_yaxis()
ax3.set_title('HTTP Status Codes', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add category labels
ax3.text(0.05, -0.5, 'Success (2xx)', fontsize=9, color=MLGREEN, fontweight='bold')
ax3.text(0.05, 2.5, 'Client Error (4xx)', fontsize=9, color=MLORANGE, fontweight='bold')
ax3.text(0.05, 5.5, 'Server Error (5xx)', fontsize=9, color=MLRED, fontweight='bold')

# Plot 4: Response format
ax4 = axes[1, 1]
ax4.axis('off')

response_format = '''
RESPONSE FORMAT BEST PRACTICES

CONSISTENT STRUCTURE:
---------------------
Always return same structure!

# Success response
{
    "success": true,
    "data": {
        "prediction": "buy",
        "confidence": 0.85,
        "model_version": "2.0.0"
    },
    "timestamp": "2024-01-15T14:30:00Z"
}

# Error response
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid feature: price must be positive",
        "details": {"field": "price", "value": -10}
    },
    "timestamp": "2024-01-15T14:30:00Z"
}


INCLUDE METADATA:
-----------------
{
    "data": {...},
    "meta": {
        "model_version": "2.0.0",
        "processing_time_ms": 45,
        "request_id": "abc123"
    }
}


PAGINATION FOR LISTS:
---------------------
{
    "data": [...],
    "pagination": {
        "page": 1,
        "per_page": 20,
        "total": 100,
        "total_pages": 5
    }
}


FASTAPI RESPONSE MODELS:
------------------------
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
def predict(data: Input):
    ...
'''

ax4.text(0.02, 0.98, response_format, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Response Format', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
