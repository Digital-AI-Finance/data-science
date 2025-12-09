"""FastAPI Basics - Getting Started"""
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
fig.suptitle('FastAPI Basics: Getting Started', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Installation and first app
ax1 = axes[0, 0]
ax1.axis('off')

basics = '''
FASTAPI INSTALLATION & FIRST APP

INSTALLATION:
-------------
pip install fastapi uvicorn

fastapi: The framework
uvicorn: ASGI server to run the app


MINIMAL APP (main.py):
----------------------
from fastapi import FastAPI

# Create app instance
app = FastAPI()


# Define endpoint
@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


RUN THE APP:
------------
uvicorn main:app --reload

main:   Python file (main.py)
app:    FastAPI instance name
--reload: Auto-reload on code changes


ACCESS:
-------
Browser: http://127.0.0.1:8000
Docs:    http://127.0.0.1:8000/docs


OUTPUT:
-------
{"message": "Hello, World!"}


THAT'S IT!
----------
5 lines of code = working API!
'''

ax1.text(0.02, 0.98, basics, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Installation & First App', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Path and query parameters
ax2 = axes[0, 1]
ax2.axis('off')

parameters = '''
PATH & QUERY PARAMETERS

PATH PARAMETERS:
----------------
Part of the URL path.

@app.get("/stocks/{ticker}")
def get_stock(ticker: str):
    return {"ticker": ticker}

# URL: /stocks/AAPL
# Result: {"ticker": "AAPL"}


@app.get("/stocks/{ticker}/price")
def get_price(ticker: str):
    price = get_stock_price(ticker)
    return {"ticker": ticker, "price": price}


QUERY PARAMETERS:
-----------------
After ? in URL.

@app.get("/stocks")
def search_stocks(sector: str = None, limit: int = 10):
    return {"sector": sector, "limit": limit}

# URL: /stocks?sector=tech&limit=5
# Result: {"sector": "tech", "limit": 5}


COMBINING BOTH:
---------------
@app.get("/stocks/{ticker}/history")
def get_history(
    ticker: str,           # Path parameter
    days: int = 30,        # Query parameter (default=30)
    include_volume: bool = False  # Query parameter
):
    return {
        "ticker": ticker,
        "days": days,
        "include_volume": include_volume
    }

# URL: /stocks/AAPL/history?days=60&include_volume=true


TYPE VALIDATION:
----------------
FastAPI automatically validates types!

# If user sends: /stocks/AAPL/history?days=abc
# Error: {"detail": "value is not a valid integer"}
'''

ax2.text(0.02, 0.98, parameters, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Path & Query Parameters', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Request body (POST)
ax3 = axes[1, 0]
ax3.axis('off')

post_request = '''
REQUEST BODY (POST REQUESTS)

For sending complex data, use POST with request body.


BASIC POST:
-----------
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Define data structure
class StockData(BaseModel):
    ticker: str
    price: float
    volume: int


@app.post("/stocks/create")
def create_stock(stock: StockData):
    # stock is automatically validated
    return {"received": stock.dict()}


# Send via curl:
# curl -X POST "http://127.0.0.1:8000/stocks/create" \\
#      -H "Content-Type: application/json" \\
#      -d '{"ticker": "AAPL", "price": 150.5, "volume": 1000}'


# Or via Python requests:
import requests

data = {"ticker": "AAPL", "price": 150.5, "volume": 1000}
response = requests.post(
    "http://127.0.0.1:8000/stocks/create",
    json=data
)
print(response.json())


VALIDATION HAPPENS AUTOMATICALLY:
---------------------------------
If price is not a float -> Error
If ticker is missing -> Error
If extra field sent -> Ignored (by default)


RESPONSE:
---------
{
    "received": {
        "ticker": "AAPL",
        "price": 150.5,
        "volume": 1000
    }
}
'''

ax3.text(0.02, 0.98, post_request, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Request Body (POST)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: HTTP methods visualization
ax4 = axes[1, 1]
ax4.axis('off')

# Create table of HTTP methods
methods = [
    ['GET', 'Retrieve data', '@app.get("/items")', 'Read-only'],
    ['POST', 'Create data', '@app.post("/items")', 'Send body'],
    ['PUT', 'Update data', '@app.put("/items/{id}")', 'Full update'],
    ['PATCH', 'Partial update', '@app.patch("/items/{id}")', 'Partial'],
    ['DELETE', 'Remove data', '@app.delete("/items/{id}")', 'Remove']
]

columns = ['Method', 'Purpose', 'Decorator', 'Notes']

table = ax4.table(cellText=methods, colLabels=columns,
                  loc='center', cellLoc='center',
                  colColours=[MLLAVENDER]*4)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2)

# Color code methods
colors = {'GET': MLBLUE, 'POST': MLGREEN, 'PUT': MLORANGE, 'PATCH': MLPURPLE, 'DELETE': MLRED}
for i, method in enumerate(['GET', 'POST', 'PUT', 'PATCH', 'DELETE']):
    table[i+1, 0].set_facecolor(colors[method])
    table[i+1, 0].set_text_props(color='white', fontweight='bold')

ax4.axis('off')
ax4.set_title('HTTP Methods', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add note
ax4.text(0.5, 0.15, 'For ML predictions: Usually POST (sending feature data)',
         fontsize=10, ha='center', style='italic',
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
