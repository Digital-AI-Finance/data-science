"""Swagger Documentation - Auto-Generated API Docs"""
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
fig.suptitle('Swagger Documentation: Auto-Generated API Docs', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is Swagger
ax1 = axes[0, 0]
ax1.axis('off')

swagger_intro = '''
WHAT IS SWAGGER / OPENAPI?

DEFINITION:
-----------
OpenAPI (formerly Swagger) is a specification
for documenting REST APIs.

FastAPI generates this AUTOMATICALLY!


ACCESS DOCS:
------------
http://localhost:8000/docs     <- Swagger UI
http://localhost:8000/redoc    <- ReDoc (alternative)
http://localhost:8000/openapi.json  <- Raw spec


WHAT YOU GET FOR FREE:
----------------------
- Interactive documentation
- API testing interface
- Request/response examples
- Schema validation
- Try-it-out functionality


SWAGGER UI FEATURES:
--------------------
1. List all endpoints
2. Show request/response schemas
3. Execute requests directly
4. View response codes
5. Authentication support


NO EXTRA CODE NEEDED:
---------------------
Just define your endpoints with type hints,
and FastAPI generates everything!

@app.post("/predict")
def predict(data: StockInput) -> PredictionOutput:
    ...

This creates:
- Endpoint documentation
- Input schema from StockInput
- Output schema from PredictionOutput
- Example values
- Try-it interface
'''

ax1.text(0.02, 0.98, swagger_intro, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is Swagger?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Adding documentation
ax2 = axes[0, 1]
ax2.axis('off')

adding_docs = '''
ENHANCING DOCUMENTATION

APP METADATA:
-------------
app = FastAPI(
    title="Stock Prediction API",
    description=\"\"\"
    ML-powered stock prediction service.

    ## Features
    - Single stock predictions
    - Batch predictions
    - Model information

    ## Authentication
    API key required in header.
    \"\"\",
    version="2.0.0",
    contact={
        "name": "Data Science Team",
        "email": "ds@company.com"
    },
    license_info={
        "name": "MIT"
    }
)


ENDPOINT DOCUMENTATION:
-----------------------
@app.post(
    "/predict",
    summary="Make a prediction",
    description="Returns buy/sell signal for given stock features",
    response_description="Prediction with confidence score",
    tags=["predictions"]
)
def predict(data: StockInput) -> PredictionOutput:
    \"\"\"
    Make a stock direction prediction.

    - **price**: Current stock price (must be positive)
    - **volume**: Trading volume
    - **momentum**: Price momentum indicator

    Returns prediction with confidence score.
    \"\"\"
    return model.predict(data)


SCHEMA EXAMPLES:
----------------
class StockInput(BaseModel):
    price: float
    volume: int

    class Config:
        schema_extra = {
            "example": {
                "price": 150.5,
                "volume": 1000000
            }
        }
'''

ax2.text(0.02, 0.98, adding_docs, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Adding Documentation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Swagger UI mockup
ax3 = axes[1, 0]
ax3.axis('off')

# Draw Swagger UI mockup
# Header
ax3.add_patch(plt.Rectangle((0.02, 0.88), 0.96, 0.1, facecolor=MLGREEN, alpha=0.8))
ax3.text(0.5, 0.93, 'Stock Prediction API v2.0.0', fontsize=12, ha='center',
         color='white', fontweight='bold')

# Endpoints section
endpoints = [
    ('POST', '/predict', 'Make a prediction', MLGREEN),
    ('POST', '/predict/batch', 'Batch predictions', MLGREEN),
    ('GET', '/model/info', 'Get model information', MLBLUE),
    ('GET', '/health', 'Health check', MLBLUE)
]

y = 0.8
for method, path, desc, color in endpoints:
    ax3.add_patch(plt.Rectangle((0.05, y-0.08), 0.1, 0.06, facecolor=color, alpha=0.8))
    ax3.text(0.1, y-0.05, method, fontsize=8, ha='center', color='white', fontweight='bold')
    ax3.text(0.17, y-0.05, path, fontsize=9, fontweight='bold')
    ax3.text(0.5, y-0.05, desc, fontsize=8, color='gray')
    y -= 0.12

# Try it out section
ax3.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.25, facecolor='white', edgecolor='gray'))
ax3.text(0.1, 0.32, 'Try it out', fontsize=10, fontweight='bold')
ax3.text(0.1, 0.25, 'Request body:', fontsize=9)
ax3.text(0.1, 0.18, '{"price": 150.5, "volume": 1000000}', fontsize=8, fontfamily='monospace')
ax3.add_patch(plt.Rectangle((0.7, 0.12), 0.2, 0.06, facecolor=MLBLUE, alpha=0.8))
ax3.text(0.8, 0.15, 'Execute', fontsize=9, ha='center', color='white')

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title('Swagger UI Preview', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Tags and grouping
ax4 = axes[1, 1]
ax4.axis('off')

tags_code = '''
ORGANIZING WITH TAGS

DEFINE TAGS:
------------
from fastapi import FastAPI

tags_metadata = [
    {
        "name": "predictions",
        "description": "Make stock predictions"
    },
    {
        "name": "model",
        "description": "Model information and health"
    },
    {
        "name": "admin",
        "description": "Administrative endpoints"
    }
]

app = FastAPI(openapi_tags=tags_metadata)


USE TAGS ON ENDPOINTS:
----------------------
@app.post("/predict", tags=["predictions"])
def predict(data: StockInput):
    ...

@app.post("/predict/batch", tags=["predictions"])
def predict_batch(data: List[StockInput]):
    ...

@app.get("/model/info", tags=["model"])
def model_info():
    ...

@app.get("/health", tags=["model"])
def health():
    ...


RESULT IN SWAGGER UI:
---------------------
predictions
  POST /predict
  POST /predict/batch

model
  GET /model/info
  GET /health

admin
  (admin endpoints here)


BENEFITS:
---------
- Organized documentation
- Easy to navigate
- Group related endpoints
- Professional appearance
'''

ax4.text(0.02, 0.98, tags_code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Tags and Grouping', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
