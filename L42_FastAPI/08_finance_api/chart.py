"""Finance API - Complete Stock Prediction Service"""
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
fig.suptitle('Finance API: Complete Stock Prediction Service', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Complete finance API
ax1 = axes[0, 0]
ax1.axis('off')

complete_api = '''
COMPLETE FINANCE API (main.py)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
from datetime import datetime


# Load models at startup
model = joblib.load('models/stock_classifier.joblib')
features = joblib.load('models/features.joblib')


# Schemas
class StockFeatures(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5)
    price: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    momentum: float = 0.0
    volatility: float = 0.0
    rsi: float = Field(default=50, ge=0, le=100)

    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL",
                "price": 150.5,
                "volume": 1000000,
                "momentum": 0.02,
                "volatility": 0.15,
                "rsi": 55
            }
        }


class PredictionResult(BaseModel):
    ticker: str
    signal: str  # "buy", "sell", "hold"
    confidence: float
    timestamp: datetime


# Create app
app = FastAPI(
    title="Stock Prediction API",
    version="2.0.0"
)


@app.post("/api/v1/predict")
def predict(stock: StockFeatures) -> PredictionResult:
    X = np.array([[stock.price, stock.volume,
                   stock.momentum, stock.volatility, stock.rsi]])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    signals = {0: "sell", 1: "hold", 2: "buy"}
    return PredictionResult(
        ticker=stock.ticker,
        signal=signals[pred],
        confidence=float(max(proba)),
        timestamp=datetime.now()
    )
'''

ax1.text(0.02, 0.98, complete_api, transform=ax1.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Complete Finance API', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Additional finance endpoints
ax2 = axes[0, 1]
ax2.axis('off')

additional = '''
ADDITIONAL FINANCE ENDPOINTS

# Portfolio prediction
class PortfolioInput(BaseModel):
    stocks: List[StockFeatures]


class PortfolioResult(BaseModel):
    predictions: List[PredictionResult]
    overall_signal: str
    buy_count: int
    sell_count: int


@app.post("/api/v1/portfolio/predict")
def predict_portfolio(portfolio: PortfolioInput) -> PortfolioResult:
    predictions = [predict(stock) for stock in portfolio.stocks]

    buy_count = sum(1 for p in predictions if p.signal == "buy")
    sell_count = sum(1 for p in predictions if p.signal == "sell")

    if buy_count > sell_count:
        overall = "bullish"
    elif sell_count > buy_count:
        overall = "bearish"
    else:
        overall = "neutral"

    return PortfolioResult(
        predictions=predictions,
        overall_signal=overall,
        buy_count=buy_count,
        sell_count=sell_count
    )


# Historical predictions (with caching)
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_prediction(ticker: str, price: float, volume: int):
    return predict(StockFeatures(
        ticker=ticker, price=price, volume=volume
    ))


# Model metrics endpoint
@app.get("/api/v1/model/metrics")
def get_model_metrics():
    return {
        "accuracy": 0.89,
        "precision": 0.87,
        "recall": 0.86,
        "last_trained": "2024-01-15",
        "training_samples": 50000
    }
'''

ax2.text(0.02, 0.98, additional, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Additional Endpoints', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Project structure
ax3 = axes[1, 0]
ax3.axis('off')

structure = '''
PROJECT STRUCTURE

stock_prediction_api/
|
+-- main.py                 # FastAPI application
+-- schemas.py              # Pydantic models
+-- predict.py              # Prediction logic
|
+-- models/
|   +-- stock_classifier.joblib
|   +-- features.joblib
|   +-- metadata.json
|
+-- routers/
|   +-- predictions.py      # /predict endpoints
|   +-- model.py            # /model endpoints
|   +-- health.py           # /health endpoint
|
+-- tests/
|   +-- test_predictions.py
|   +-- test_api.py
|
+-- requirements.txt
+-- Dockerfile
+-- README.md


requirements.txt:
-----------------
fastapi==0.104.0
uvicorn==0.24.0
pydantic==2.5.0
joblib==1.3.2
scikit-learn==1.3.0
numpy==1.24.3


RUN LOCALLY:
------------
uvicorn main:app --reload --port 8000


RUN WITH DOCKER:
----------------
docker build -t stock-api .
docker run -p 8000:8000 stock-api
'''

ax3.text(0.02, 0.98, structure, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Project Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: API usage dashboard
ax4 = axes[1, 1]

# Simulated API usage stats
hours = np.arange(24)
requests = np.array([50, 30, 20, 15, 10, 20, 100, 300, 450, 500, 480, 520,
                     550, 530, 490, 460, 400, 350, 200, 150, 120, 90, 70, 60])

ax4.fill_between(hours, requests, alpha=0.3, color=MLBLUE)
ax4.plot(hours, requests, color=MLBLUE, linewidth=2)

ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('API Requests')
ax4.set_title('API Usage Pattern (Example)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3)

# Add statistics
stats_text = f'''Daily Stats:
Total: {sum(requests):,}
Peak: {max(requests)}
Avg: {np.mean(requests):.0f}'''
ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Mark peak hours
peak_start, peak_end = 8, 16
ax4.axvspan(peak_start, peak_end, alpha=0.1, color=MLGREEN)
ax4.text(12, 50, 'Trading Hours', fontsize=9, ha='center', color=MLGREEN)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
