"""Error Handling in FastAPI"""
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
fig.suptitle('Error Handling in FastAPI', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: HTTP exceptions
ax1 = axes[0, 0]
ax1.axis('off')

exceptions = '''
HTTP EXCEPTIONS

BASIC HTTP EXCEPTION:
---------------------
from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.post("/predict")
def predict(data: StockInput):
    # Check if ticker is valid
    if data.ticker not in valid_tickers:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker {data.ticker} not found"
        )

    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )

    return model.predict(data)


COMMON STATUS CODES:
--------------------
400 - Bad Request (invalid input)
401 - Unauthorized (auth required)
403 - Forbidden (no permission)
404 - Not Found (resource missing)
422 - Validation Error (wrong types)
500 - Server Error (your code crashed)
503 - Service Unavailable (model down)


WITH HEADERS:
-------------
raise HTTPException(
    status_code=401,
    detail="Invalid API key",
    headers={"WWW-Authenticate": "Bearer"}
)


AUTOMATIC VALIDATION ERRORS:
----------------------------
FastAPI returns 422 automatically when
Pydantic validation fails.

{
    "detail": [{
        "loc": ["body", "price"],
        "msg": "value is not a valid float",
        "type": "type_error.float"
    }]
}
'''

ax1.text(0.02, 0.98, exceptions, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('HTTP Exceptions', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Custom exception handlers
ax2 = axes[0, 1]
ax2.axis('off')

custom_handlers = '''
CUSTOM EXCEPTION HANDLERS

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# Custom exception class
class ModelNotLoadedError(Exception):
    def __init__(self, model_name: str):
        self.model_name = model_name


class PredictionError(Exception):
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details


app = FastAPI()


# Register exception handlers
@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=503,
        content={
            "error": "MODEL_NOT_LOADED",
            "message": f"Model '{exc.model_name}' is not available",
            "suggestion": "Try again later or contact support"
        }
    )


@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc: PredictionError):
    return JSONResponse(
        status_code=500,
        content={
            "error": "PREDICTION_FAILED",
            "message": exc.message,
            "details": exc.details
        }
    )


# Use in endpoints
@app.post("/predict")
def predict(data: StockInput):
    if model is None:
        raise ModelNotLoadedError("stock_classifier")

    try:
        result = model.predict(data)
    except Exception as e:
        raise PredictionError(
            message="Model prediction failed",
            details={"error": str(e)}
        )

    return result
'''

ax2.text(0.02, 0.98, custom_handlers, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Custom Exception Handlers', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Error response structure
ax3 = axes[1, 0]
ax3.axis('off')

error_structure = '''
STANDARDIZED ERROR RESPONSES

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ErrorDetail(BaseModel):
    field: str
    message: str
    code: str


class ErrorResponse(BaseModel):
    success: bool = False
    error_code: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime
    request_id: Optional[str] = None


# Example error responses:

# Validation error
{
    "success": false,
    "error_code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
        {"field": "price", "message": "must be positive", "code": "GT_0"},
        {"field": "volume", "message": "required field", "code": "REQUIRED"}
    ],
    "timestamp": "2024-01-15T14:30:00Z",
    "request_id": "abc123"
}

# Model error
{
    "success": false,
    "error_code": "MODEL_ERROR",
    "message": "Prediction failed",
    "details": null,
    "timestamp": "2024-01-15T14:30:00Z",
    "request_id": "abc123"
}

# Auth error
{
    "success": false,
    "error_code": "UNAUTHORIZED",
    "message": "Invalid API key",
    "details": null,
    "timestamp": "2024-01-15T14:30:00Z"
}


CONSISTENCY IS KEY:
-------------------
Always return the same structure!
Clients can handle errors uniformly.
'''

ax3.text(0.02, 0.98, error_structure, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Error Response Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Error handling best practices
ax4 = axes[1, 1]

# Create checklist visualization
practices = [
    ('Validate input early', 'Use Pydantic schemas'),
    ('Catch model errors', 'Wrap predict() in try/except'),
    ('Log all errors', 'Use proper logging'),
    ('Return clear messages', 'Help users fix issues'),
    ('Use status codes', 'Follow HTTP standards'),
    ('Include request ID', 'For debugging')
]

y_positions = np.arange(len(practices))

for i, (practice, detail) in enumerate(practices):
    ax4.add_patch(plt.Rectangle((0.05, i-0.3), 0.9, 0.6, facecolor=MLGREEN, alpha=0.2))
    ax4.text(0.1, i, f"[OK] {practice}", fontsize=10, va='center', fontweight='bold')
    ax4.text(0.55, i, detail, fontsize=9, va='center', style='italic')

ax4.set_ylim(-0.5, len(practices)-0.5)
ax4.set_xlim(0, 1)
ax4.invert_yaxis()
ax4.axis('off')
ax4.set_title('Best Practices Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
