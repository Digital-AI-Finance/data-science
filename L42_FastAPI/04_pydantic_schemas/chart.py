"""Pydantic Schemas - Data Validation"""
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
fig.suptitle('Pydantic Schemas: Data Validation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is Pydantic
ax1 = axes[0, 0]
ax1.axis('off')

pydantic_intro = '''
WHAT IS PYDANTIC?

DEFINITION:
-----------
Data validation library using Python type hints.
Automatically validates incoming data.


WHY USE PYDANTIC?
-----------------
- Automatic type checking
- Clear error messages
- Data conversion
- Documentation generation
- Works seamlessly with FastAPI


BASIC EXAMPLE:
--------------
from pydantic import BaseModel

class StockInput(BaseModel):
    ticker: str
    price: float
    volume: int

# Valid data
data = StockInput(ticker="AAPL", price=150.5, volume=1000)
# Works!

# Invalid data
data = StockInput(ticker="AAPL", price="not_a_number", volume=1000)
# Error: value is not a valid float


AUTOMATIC CONVERSION:
---------------------
class Example(BaseModel):
    value: int

Example(value="123")  # Converts string to int!
Example(value=123.9)  # Converts to 123


THIS PREVENTS:
--------------
- Invalid data reaching your model
- Runtime crashes from bad input
- Security issues
- Hard-to-debug errors
'''

ax1.text(0.02, 0.98, pydantic_intro, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is Pydantic?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Schema definitions
ax2 = axes[0, 1]
ax2.axis('off')

schemas = '''
DEFINING SCHEMAS FOR ML

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# INPUT SCHEMA
class PredictionInput(BaseModel):
    price: float = Field(..., gt=0, description="Stock price")
    volume: int = Field(..., ge=0, description="Trading volume")
    momentum: float = Field(default=0.0, description="Price momentum")
    ticker: str = Field(..., min_length=1, max_length=5)

    class Config:
        schema_extra = {
            "example": {
                "price": 150.5,
                "volume": 1000000,
                "momentum": 0.02,
                "ticker": "AAPL"
            }
        }


# OUTPUT SCHEMA
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float = Field(..., ge=0, le=1)
    model_version: str
    timestamp: datetime


# BATCH INPUT
class BatchPredictionInput(BaseModel):
    stocks: List[PredictionInput]


# ERROR RESPONSE
class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[dict] = None


FIELD VALIDATORS:
-----------------
gt:  greater than
ge:  greater than or equal
lt:  less than
le:  less than or equal
min_length, max_length: for strings
regex: pattern matching
'''

ax2.text(0.02, 0.98, schemas, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Schema Definitions', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Custom validators
ax3 = axes[1, 0]
ax3.axis('off')

validators = '''
CUSTOM VALIDATORS

from pydantic import BaseModel, validator, root_validator


class StockPredictionInput(BaseModel):
    ticker: str
    price: float
    volume: int
    date: str


    # Field validator
    @validator('ticker')
    def ticker_must_be_uppercase(cls, v):
        if not v.isupper():
            raise ValueError('ticker must be uppercase')
        return v


    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('price must be positive')
        return v


    @validator('date')
    def validate_date_format(cls, v):
        from datetime import datetime
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('date must be YYYY-MM-DD format')
        return v


    # Root validator (checks multiple fields)
    @root_validator
    def check_volume_price_ratio(cls, values):
        price = values.get('price')
        volume = values.get('volume')
        if price and volume:
            if volume / price > 1000000:
                raise ValueError('suspicious volume/price ratio')
        return values


# USAGE
try:
    data = StockPredictionInput(
        ticker="aapl",  # Will fail - not uppercase
        price=150.5,
        volume=1000,
        date="2024-01-15"
    )
except ValueError as e:
    print(e)  # "ticker must be uppercase"
'''

ax3.text(0.02, 0.98, validators, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Custom Validators', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Using schemas in FastAPI
ax4 = axes[1, 1]
ax4.axis('off')

fastapi_usage = '''
USING SCHEMAS IN FASTAPI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


# Define schemas
class PredictionInput(BaseModel):
    ticker: str
    price: float
    volume: int


class PredictionOutput(BaseModel):
    ticker: str
    prediction: str
    confidence: float


app = FastAPI()


# Use as request body
@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    # data is already validated!
    result = model.predict([[data.price, data.volume]])

    return PredictionOutput(
        ticker=data.ticker,
        prediction="buy" if result[0] == 1 else "sell",
        confidence=0.85
    )


# Batch endpoint
@app.post("/predict/batch", response_model=List[PredictionOutput])
def predict_batch(data: List[PredictionInput]):
    results = []
    for item in data:
        pred = predict(item)
        results.append(pred)
    return results


# What you get automatically:
# - Input validation
# - Output validation
# - Auto-generated documentation
# - Clear error messages
# - Type hints for IDE support


# Error example (automatic):
# POST /predict with {"ticker": "AAPL", "price": "bad"}
# Response 422:
# {
#   "detail": [{
#     "loc": ["body", "price"],
#     "msg": "value is not a valid float",
#     "type": "type_error.float"
#   }]
# }
'''

ax4.text(0.02, 0.98, fastapi_usage, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Using Schemas in FastAPI', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
