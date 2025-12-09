"""Live Deployment - Putting It All Together"""
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
fig.suptitle('Live Deployment: Putting It All Together', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Final project structure
ax1 = axes[0, 0]
ax1.axis('off')

structure = '''
FINAL PROJECT STRUCTURE

stock-dashboard/
|
+-- app.py                  # Main Streamlit app
|
+-- requirements.txt        # Dependencies
|   streamlit==1.29.0
|   pandas==2.0.3
|   numpy==1.24.3
|   scikit-learn==1.3.0
|   joblib==1.3.2
|   plotly==5.18.0
|
+-- .streamlit/
|   +-- config.toml        # Theme settings
|
+-- models/
|   +-- stock_classifier.joblib
|   +-- features.joblib
|
+-- data/
|   +-- sample_stocks.csv  # Small sample data
|
+-- .gitignore             # Excludes secrets
|   .streamlit/secrets.toml
|   __pycache__/
|   *.pyc
|
+-- README.md              # Documentation
|
+-- LICENSE                # MIT License


READY FOR DEPLOYMENT!
---------------------
All files in place.
Push to GitHub.
Deploy on Streamlit Cloud.
'''

ax1.text(0.02, 0.98, structure, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Final Project Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Deployment commands summary
ax2 = axes[0, 1]
ax2.axis('off')

commands = '''
DEPLOYMENT COMMANDS SUMMARY

1. PREPARE LOCAL ENVIRONMENT:
-----------------------------
# Create virtual environment
python -m venv venv
venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Test app
streamlit run app.py


2. INITIALIZE GIT:
------------------
git init
git add .
git commit -m "Initial commit: Stock Dashboard"


3. PUSH TO GITHUB:
------------------
# Create repo on GitHub first, then:
git remote add origin https://github.com/USER/stock-dashboard.git
git branch -M main
git push -u origin main


4. DEPLOY ON STREAMLIT CLOUD:
-----------------------------
# Go to share.streamlit.io
# Click "New app"
# Select: USER/stock-dashboard / main / app.py
# Click "Deploy!"


5. CONFIGURE SECRETS (if needed):
---------------------------------
# In Streamlit Cloud: Settings -> Secrets
API_KEY = "your-key"

[database]
host = "db.example.com"


6. UPDATE YOUR APP:
-------------------
# Make changes locally
# Test: streamlit run app.py
# Commit: git add . && git commit -m "Update"
# Push: git push
# App auto-redeploys!


YOUR APP IS LIVE!
-----------------
https://user-stock-dashboard-main-app.streamlit.app
'''

ax2.text(0.02, 0.98, commands, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Commands Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Success visualization
ax3 = axes[1, 0]
ax3.axis('off')

# Draw success celebration
ax3.text(0.5, 0.9, 'DEPLOYMENT SUCCESS!', fontsize=14, ha='center',
         fontweight='bold', color=MLGREEN)

# App preview box
ax3.add_patch(plt.Rectangle((0.15, 0.25), 0.7, 0.55, facecolor='white', edgecolor=MLBLUE, linewidth=2))

# Browser bar
ax3.add_patch(plt.Rectangle((0.15, 0.72), 0.7, 0.08, facecolor=MLLAVENDER))
ax3.text(0.5, 0.76, 'user-stock-dashboard-main-app.streamlit.app', fontsize=8, ha='center')

# App content
ax3.text(0.5, 0.65, 'Stock Analysis Dashboard', fontsize=11, ha='center', fontweight='bold')

# Mini metrics
for i, (label, val) in enumerate([('Price', '$150'), ('Change', '+2.5%'), ('Volume', '1.2M')]):
    x = 0.25 + i * 0.2
    ax3.add_patch(plt.Rectangle((x, 0.5), 0.15, 0.1, facecolor=MLGREEN, alpha=0.3))
    ax3.text(x + 0.075, 0.57, label, fontsize=7, ha='center')
    ax3.text(x + 0.075, 0.52, val, fontsize=9, ha='center', fontweight='bold')

# Mini chart
x_chart = np.linspace(0.2, 0.8, 30)
y_chart = 0.38 + 0.05 * np.sin(np.linspace(0, 6, 30)) + 0.03 * np.cumsum(np.random.randn(30) * 0.1)
ax3.plot(x_chart, y_chart, color=MLBLUE, linewidth=2)

# Share button
ax3.add_patch(plt.Rectangle((0.35, 0.12), 0.3, 0.08, facecolor=MLPURPLE, alpha=0.8))
ax3.text(0.5, 0.16, 'Share URL', fontsize=10, ha='center', color='white')

ax3.set_xlim(0, 1)
ax3.set_ylim(0.05, 1)
ax3.set_title('Your Live App!', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Next steps
ax4 = axes[1, 1]
ax4.axis('off')

next_steps = '''
NEXT STEPS AFTER DEPLOYMENT

SHARE YOUR APP:
---------------
- Copy URL from Streamlit Cloud
- Share with colleagues, instructors
- Add to your portfolio/resume
- Post on LinkedIn!


GATHER FEEDBACK:
----------------
- Ask users to test
- Note any errors or confusion
- Collect feature requests


ITERATE AND IMPROVE:
--------------------
- Fix bugs reported
- Add requested features
- Improve UI/UX
- Update documentation


SHOWCASE IN PORTFOLIO:
----------------------
- Take screenshots
- Record demo video
- Write project description
- Highlight technologies used


FOR YOUR FINAL PROJECT:
-----------------------
1. Deploy early (don't wait until last day!)
2. Test thoroughly on cloud
3. Get feedback before presentation
4. Prepare backup (local demo)
5. Document your deployment process


CONGRATULATIONS!
----------------
You've learned to:
+ Build data apps with Streamlit
+ Deploy to the cloud
+ Manage secrets securely
+ Use version control
+ Monitor production apps

These are REAL industry skills!
'''

ax4.text(0.02, 0.98, next_steps, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Next Steps', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
