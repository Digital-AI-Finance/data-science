"""Code Cleanup - Preparing Your Code for Submission"""
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
fig.suptitle('Code Cleanup: Preparing Your Code for Submission', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Before vs After cleanup
ax1 = axes[0, 0]
ax1.axis('off')

comparison = '''
CODE CLEANUP: BEFORE & AFTER

BEFORE (MESSY):
---------------
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
#from sklearn.linear_model import LogisticRegression
import joblib
# import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/john/desktop/data.csv")
print(df.head())  # debugging
x = df[['a','b','c']]
# x = df[['a','b']]
y=df['target']
model = joblib.load("model.joblib")
pred=model.predict(x)
print(pred)  # check this


AFTER (CLEAN):
--------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load model and data
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models/model.joblib"
    return joblib.load(model_path)

@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "data/sample.csv"
    return pd.read_csv(data_path)

# Main app
model = load_model()
df = load_data()

# Feature selection
features = ['feature_a', 'feature_b', 'feature_c']
X = df[features]

# Prediction
prediction = model.predict(X)
'''

ax1.text(0.02, 0.98, comparison, transform=ax1.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Before & After Cleanup', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Cleanup checklist
ax2 = axes[0, 1]

checklist = [
    'Remove commented-out code',
    'Remove print() statements',
    'Fix hardcoded paths',
    'Organize imports (std, third-party, local)',
    'Add meaningful variable names',
    'Remove unused imports',
    'Add function docstrings',
    'Consistent formatting',
    'No credentials in code',
    'Error handling added'
]

status_before = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # Before cleanup
status_after = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # After cleanup

y_pos = np.arange(len(checklist))
width = 0.35

bars1 = ax2.barh(y_pos - width/2, status_before, width, label='Before', color=MLRED, alpha=0.7)
bars2 = ax2.barh(y_pos + width/2, status_after, width, label='After', color=MLGREEN, alpha=0.7)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(checklist, fontsize=8)
ax2.set_xlabel('Complete (0=No, 1=Yes)')
ax2.set_xlim(0, 1.2)
ax2.legend(fontsize=8)
ax2.set_title('Cleanup Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: File organization
ax3 = axes[1, 0]
ax3.axis('off')

organization = '''
FILE ORGANIZATION

RECOMMENDED STRUCTURE:
----------------------
project-name/
|
+-- app.py              # Main Streamlit app
|
+-- requirements.txt    # Dependencies with versions
|
+-- README.md           # Project documentation
|
+-- .gitignore          # Exclude secrets, cache
|
+-- models/
|   +-- classifier.joblib
|   +-- scaler.joblib
|
+-- data/
|   +-- sample.csv      # Small sample for demo
|
+-- notebooks/          # (optional) EDA notebooks
|   +-- exploration.ipynb
|
+-- .streamlit/
|   +-- config.toml     # Theme settings


.GITIGNORE CONTENTS:
--------------------
# Secrets
.streamlit/secrets.toml
.env
*.pem

# Python
__pycache__/
*.pyc
.Python
venv/

# IDE
.vscode/
.idea/

# Data (if large)
data/raw/
*.csv.gz


REQUIREMENTS.TXT:
-----------------
streamlit==1.29.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
plotly==5.18.0
'''

ax3.text(0.02, 0.98, organization, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('File Organization', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Common issues to fix
ax4 = axes[1, 1]
ax4.axis('off')

issues = '''
COMMON ISSUES TO FIX

PATHS:
------
BAD:  pd.read_csv("C:/Users/me/data.csv")
GOOD: pd.read_csv(Path(__file__).parent / "data/sample.csv")


IMPORTS:
--------
BAD:  from sklearn.ensemble import *
GOOD: from sklearn.ensemble import RandomForestClassifier


VARIABLES:
----------
BAD:  x = df[['a', 'b', 'c']]
GOOD: features = df[['price', 'volume', 'returns']]


ERROR HANDLING:
---------------
BAD:  model = joblib.load("model.joblib")

GOOD: try:
          model = joblib.load("model.joblib")
      except FileNotFoundError:
          st.error("Model file not found!")
          st.stop()


CACHING:
--------
BAD:  def load_data():
          return pd.read_csv("data.csv")

GOOD: @st.cache_data
      def load_data():
          return pd.read_csv("data.csv")


MAGIC NUMBERS:
--------------
BAD:  if prediction > 0.7:
GOOD: CONFIDENCE_THRESHOLD = 0.7
      if prediction > CONFIDENCE_THRESHOLD:


FINAL CHECK:
------------
- Run: python -m py_compile app.py
- Run: streamlit run app.py
- Test all features once
'''

ax4.text(0.02, 0.98, issues, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Common Issues to Fix', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
