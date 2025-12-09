"""API Concept - Why Build APIs for ML Models"""
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
fig.suptitle('API Concept: Why Build APIs for ML Models?', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is an API
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT IS AN API?

DEFINITION:
-----------
API = Application Programming Interface

A way for different software applications
to communicate with each other.


REST API:
---------
Most common type for web services.
Uses HTTP requests (GET, POST, PUT, DELETE).


FOR ML MODELS:
--------------
API allows your model to:
- Receive data from any client
- Return predictions
- Be used by web apps, mobile apps, other services


EXAMPLE FLOW:
-------------
Client -> HTTP Request -> API Server -> Model -> Prediction
                                    <-   Response   <-


WHY NOT JUST RUN PYTHON?
------------------------
- Python script = one user, one machine
- API = many users, any platform
- API = scalable, production-ready


KEY CONCEPTS:
-------------
Endpoint: URL where requests are sent
  /api/predict

Request: Data sent to the API
  {"features": [1.2, 3.4, 5.6]}

Response: Data returned from API
  {"prediction": "buy", "confidence": 0.85}


HTTP METHODS:
-------------
GET:    Retrieve data
POST:   Send data (most common for ML)
PUT:    Update data
DELETE: Remove data
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is an API?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual architecture
ax2 = axes[0, 1]
ax2.axis('off')

# Draw API architecture
# Clients
clients = [('Web App', 0.1, 0.8), ('Mobile App', 0.1, 0.5), ('Other Service', 0.1, 0.2)]
for name, x, y in clients:
    ax2.add_patch(plt.Rectangle((x-0.08, y-0.08), 0.16, 0.12, facecolor=MLBLUE, alpha=0.3))
    ax2.text(x, y, name, fontsize=8, ha='center', va='center')

# Arrows to API
for _, x, y in clients:
    ax2.annotate('', xy=(0.35, 0.5), xytext=(x+0.08, y),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=1.5))

# API Server
ax2.add_patch(plt.Rectangle((0.35, 0.35), 0.2, 0.3, facecolor=MLGREEN, alpha=0.3))
ax2.text(0.45, 0.5, 'FastAPI\nServer', fontsize=10, ha='center', va='center', fontweight='bold')

# Arrow to model
ax2.annotate('', xy=(0.65, 0.5), xytext=(0.55, 0.5),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))

# ML Model
ax2.add_patch(plt.Rectangle((0.65, 0.35), 0.2, 0.3, facecolor=MLPURPLE, alpha=0.3))
ax2.text(0.75, 0.5, 'ML Model\n(joblib)', fontsize=10, ha='center', va='center', fontweight='bold')

# Labels
ax2.text(0.1, 0.95, 'CLIENTS', fontsize=10, ha='center', fontweight='bold', color=MLBLUE)
ax2.text(0.45, 0.72, 'API', fontsize=10, ha='center', fontweight='bold', color=MLGREEN)
ax2.text(0.75, 0.72, 'MODEL', fontsize=10, ha='center', fontweight='bold', color=MLPURPLE)

# Request/Response labels
ax2.text(0.28, 0.6, 'HTTP\nRequest', fontsize=7, ha='center', color=MLORANGE)
ax2.text(0.6, 0.55, 'predict()', fontsize=7, ha='center', color=MLORANGE)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('API Architecture', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: API frameworks
ax3 = axes[1, 0]
ax3.axis('off')

frameworks = '''
PYTHON API FRAMEWORKS

FLASK:
------
- Lightweight, flexible
- Good for small projects
- More manual setup

from flask import Flask, request
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return {'prediction': model.predict(data)}


FASTAPI (RECOMMENDED):
----------------------
- Modern, fast, async
- Automatic documentation
- Type hints = validation

from fastapi import FastAPI
app = FastAPI()

@app.post('/predict')
def predict(data: PredictionInput):
    return {'prediction': model.predict(data)}


DJANGO REST:
------------
- Full-featured
- Better for large applications
- More setup required


WHY FASTAPI?
------------
+ Automatic validation
+ Auto-generated docs (Swagger UI)
+ Async support (fast!)
+ Type hints make code clearer
+ Easy to learn

Performance (requests/second):
- Flask:   ~1000
- FastAPI: ~3000 (3x faster!)


INSTALLATION:
-------------
pip install fastapi uvicorn
'''

ax3.text(0.02, 0.98, frameworks, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Python API Frameworks', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Framework comparison
ax4 = axes[1, 1]

frameworks_names = ['Flask', 'FastAPI', 'Django\nREST']
ease_of_use = [4, 5, 3]
performance = [3, 5, 3]
features = [3, 4, 5]
documentation = [3, 5, 4]

x = np.arange(len(frameworks_names))
width = 0.2

bars1 = ax4.bar(x - 1.5*width, ease_of_use, width, label='Ease of Use', color=MLBLUE, edgecolor='black')
bars2 = ax4.bar(x - 0.5*width, performance, width, label='Performance', color=MLGREEN, edgecolor='black')
bars3 = ax4.bar(x + 0.5*width, features, width, label='Features', color=MLORANGE, edgecolor='black')
bars4 = ax4.bar(x + 1.5*width, documentation, width, label='Auto Docs', color=MLPURPLE, edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(frameworks_names, fontsize=10)
ax4.set_ylabel('Score (1-5)')
ax4.set_title('Framework Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8, loc='lower right')
ax4.set_ylim(0, 6)
ax4.grid(alpha=0.3, axis='y')

# Highlight FastAPI
ax4.annotate('Best for ML!', xy=(1, 5.3), fontsize=10, ha='center', color=MLGREEN, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
