"""Deployment Preparation - Getting Ready for Production"""
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
fig.suptitle('Deployment Preparation: Getting Ready for Production', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Project structure
ax1 = axes[0, 0]
ax1.axis('off')

structure = '''
PROJECT STRUCTURE FOR DEPLOYMENT

stock_dashboard/
|
+-- app.py                    # Main Streamlit app
+-- requirements.txt          # Dependencies
+-- .streamlit/
|   +-- config.toml          # Streamlit config
|   +-- secrets.toml         # API keys (local only!)
|
+-- models/
|   +-- stock_classifier.joblib
|   +-- features.joblib
|
+-- data/
|   +-- stocks.csv           # Sample data
|
+-- utils/
|   +-- __init__.py
|   +-- data_loader.py       # Data loading functions
|   +-- predictions.py       # Prediction functions
|
+-- pages/                   # Multi-page app
|   +-- 1_Dashboard.py
|   +-- 2_Predictions.py
|   +-- 3_About.py
|
+-- .gitignore               # Ignore secrets!
+-- README.md


IMPORTANT:
----------
.gitignore should include:
- .streamlit/secrets.toml
- __pycache__/
- *.pyc
- .env
'''

ax1.text(0.02, 0.98, structure, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Project Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Requirements.txt
ax2 = axes[0, 1]
ax2.axis('off')

requirements = '''
REQUIREMENTS.TXT

CREATING REQUIREMENTS:
----------------------
# Option 1: Manual (recommended)
# Create requirements.txt with exact versions

streamlit==1.29.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
plotly==5.18.0
matplotlib==3.7.3


# Option 2: From environment
pip freeze > requirements.txt
# Warning: includes ALL packages!


# Option 3: pipreqs (better)
pip install pipreqs
pipreqs /path/to/project
# Only includes imports used


BEST PRACTICES:
---------------
1. Pin exact versions (==)
2. Test requirements on fresh env
3. Keep minimal (only what you need)
4. Update periodically


TESTING REQUIREMENTS:
---------------------
# Create fresh virtual environment
python -m venv test_env
test_env\\Scripts\\activate

# Install requirements
pip install -r requirements.txt

# Run app
streamlit run app.py

# If it works, requirements are correct!


COMMON ISSUES:
--------------
- Missing dependency: Add to requirements
- Version conflict: Check compatibility
- Platform-specific: Use conditional deps
'''

ax2.text(0.02, 0.98, requirements, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Requirements.txt', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Configuration
ax3 = axes[1, 0]
ax3.axis('off')

config = '''
STREAMLIT CONFIGURATION

.streamlit/config.toml:
-----------------------
[theme]
primaryColor = "#3333B2"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
maxUploadSize = 200

[browser]
gatherUsageStats = false


.streamlit/secrets.toml (LOCAL ONLY!):
--------------------------------------
# API keys and secrets
API_KEY = "your-secret-key"
DATABASE_URL = "postgresql://..."

[database]
host = "localhost"
port = 5432
user = "admin"


ACCESSING SECRETS IN CODE:
--------------------------
import streamlit as st

# Access secrets
api_key = st.secrets["API_KEY"]
db_host = st.secrets["database"]["host"]


FOR CLOUD DEPLOYMENT:
---------------------
Secrets are configured in the cloud UI,
not in secrets.toml file!


PAGE CONFIG:
------------
st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)
'''

ax3.text(0.02, 0.98, config, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Configuration', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Deployment checklist
ax4 = axes[1, 1]

# Create checklist
checklist_items = [
    'App runs locally without errors',
    'requirements.txt complete',
    'No hardcoded secrets in code',
    'Secrets in .streamlit/secrets.toml',
    '.gitignore includes secrets',
    'Model files included',
    'Data files accessible',
    'Tested on fresh environment',
    'README.md documentation',
    'Git repository initialized'
]

y_positions = np.arange(len(checklist_items))

for i, item in enumerate(checklist_items):
    color = MLGREEN
    ax4.add_patch(plt.Rectangle((0.02, i-0.3), 0.05, 0.5, facecolor=color, alpha=0.5))
    ax4.text(0.04, i, '[OK]', fontsize=9, ha='center', va='center', color='white', fontweight='bold')
    ax4.text(0.1, i, item, fontsize=9, va='center')

ax4.set_ylim(-0.5, len(checklist_items)-0.5)
ax4.set_xlim(0, 1)
ax4.invert_yaxis()
ax4.axis('off')
ax4.set_title('Deployment Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
