"""Deployment Planning - Preparing for Cloud Deployment"""
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
fig.suptitle('Deployment Planning: Getting Ready for the Cloud', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Deployment requirements
ax1 = axes[0, 0]
ax1.axis('off')

requirements = '''
DEPLOYMENT REQUIREMENTS

REQUIRED FILES:
---------------
project-folder/
|
+-- app.py              # Streamlit app
|
+-- requirements.txt    # Dependencies
|
+-- models/
|   +-- model.joblib    # Trained model
|   +-- scaler.joblib   # Fitted scaler
|
+-- data/
|   +-- sample.csv      # Sample data
|
+-- .streamlit/
|   +-- config.toml     # Theme (optional)
|
+-- .gitignore          # Exclude secrets
|
+-- README.md           # Documentation


REQUIREMENTS.TXT EXAMPLE:
-------------------------
streamlit==1.29.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
plotly==5.18.0


GITHUB REPOSITORY:
------------------
- Must be PUBLIC for free Streamlit Cloud
- All code committed
- No secrets in repository!
- Model files < 100MB
'''

ax1.text(0.02, 0.98, requirements, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Deployment Requirements', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: App structure
ax2 = axes[0, 1]
ax2.axis('off')

app_structure = '''
STREAMLIT APP STRUCTURE

BASIC TEMPLATE:
---------------
import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="My Finance App",
    page_icon="$",
    layout="wide"
)

# Title
st.title("Finance Dashboard")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")

model = load_model()

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/sample.csv")

data = load_data()

# Sidebar inputs
st.sidebar.header("Settings")
option = st.sidebar.selectbox("Select:", ["A", "B"])

# Main content
st.write("Your analysis here...")

# Prediction
if st.button("Predict"):
    result = model.predict(X)
    st.success(f"Prediction: {result}")

# Charts
st.plotly_chart(fig)


KEY PATTERNS:
-------------
- @st.cache_resource for models
- @st.cache_data for data
- Sidebar for inputs
- Main area for results
'''

ax2.text(0.02, 0.98, app_structure, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('App Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Deployment checklist
ax3 = axes[1, 0]

checklist_items = [
    'App runs locally without errors',
    'requirements.txt is complete',
    'Model files included (<100MB)',
    'Sample data included',
    '.gitignore excludes secrets',
    'README.md has instructions',
    'Code is clean and commented',
    'All imports at top of file',
    'No hardcoded file paths',
    'Error handling in place'
]

status = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # All to-do
colors = [MLGREEN if s else MLRED for s in status]

y_pos = np.arange(len(checklist_items))

for i, (item, color) in enumerate(zip(checklist_items, colors)):
    ax3.add_patch(plt.Rectangle((0.02, i-0.4), 0.04, 0.8, facecolor=MLLAVENDER, edgecolor='gray'))
    ax3.text(0.08, i, item, fontsize=9, va='center')

ax3.set_xlim(0, 1)
ax3.set_ylim(-0.5, len(checklist_items)-0.5)
ax3.invert_yaxis()
ax3.axis('off')
ax3.set_title('Pre-Deployment Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Deployment flow
ax4 = axes[1, 1]
ax4.axis('off')

# Draw flow
steps = [
    ('LOCAL\nDEV', 'Develop and\ntest app', MLBLUE),
    ('GIT\nREPO', 'Push to\nGitHub', MLORANGE),
    ('STREAMLIT\nCLOUD', 'Connect and\ndeploy', MLGREEN),
    ('LIVE\nAPP', 'Share URL\nwith world!', MLPURPLE)
]

for i, (title, desc, color) in enumerate(steps):
    x = 0.15 + i * 0.23
    ax4.add_patch(plt.Circle((x, 0.65), 0.08, facecolor=color, alpha=0.3))
    ax4.text(x, 0.65, title, fontsize=9, ha='center', va='center', fontweight='bold')
    ax4.text(x, 0.45, desc, fontsize=8, ha='center')

    if i < len(steps) - 1:
        ax4.annotate('', xy=(x+0.12, 0.65), xytext=(x+0.08, 0.65),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax4.text(0.5, 0.9, 'DEPLOYMENT FLOW', fontsize=12, ha='center', fontweight='bold')

# Tips
ax4.text(0.5, 0.25, 'DEPLOYMENT TIPS:', fontsize=10, ha='center', fontweight='bold')
tips = [
    '- Deploy early, iterate often',
    '- Test locally before pushing',
    '- Check logs if app fails',
    '- Keep app simple initially'
]
for j, tip in enumerate(tips):
    ax4.text(0.5, 0.17 - j*0.05, tip, fontsize=9, ha='center')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Deployment Flow', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
