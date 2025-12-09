"""Streamlit Cloud - Step by Step Deployment"""
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
fig.suptitle('Streamlit Cloud: Step by Step Deployment', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Step by step guide
ax1 = axes[0, 0]
ax1.axis('off')

steps = '''
DEPLOYMENT STEPS

STEP 1: PREPARE YOUR CODE
-------------------------
Ensure your project has:
- app.py (or streamlit_app.py)
- requirements.txt
- All necessary data/model files

project/
  app.py
  requirements.txt
  models/
    model.joblib
  data/
    stocks.csv


STEP 2: PUSH TO GITHUB
----------------------
# Initialize git (if needed)
git init
git add .
git commit -m "Initial commit"

# Create GitHub repo and push
git remote add origin https://github.com/user/repo.git
git push -u origin main


STEP 3: GO TO STREAMLIT CLOUD
-----------------------------
1. Visit: share.streamlit.io
2. Sign in with GitHub
3. Click "New app"


STEP 4: CONFIGURE DEPLOYMENT
----------------------------
Select:
- Repository: your-username/your-repo
- Branch: main
- Main file path: app.py


STEP 5: DEPLOY!
---------------
Click "Deploy!"
Wait 2-5 minutes for build.
Your app is live!
'''

ax1.text(0.02, 0.98, steps, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Deployment Steps', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual workflow
ax2 = axes[0, 1]
ax2.axis('off')

# Draw workflow
steps_visual = [
    (0.15, 0.8, 'Local\nDevelopment', MLBLUE),
    (0.4, 0.8, 'GitHub\nRepository', MLORANGE),
    (0.65, 0.8, 'Streamlit\nCloud', MLGREEN),
    (0.9, 0.8, 'Live\nApp!', MLPURPLE)
]

for x, y, label, color in steps_visual:
    ax2.add_patch(plt.Circle((x, y), 0.1, facecolor=color, alpha=0.4))
    ax2.text(x, y, label, fontsize=9, ha='center', va='center')

# Arrows
ax2.annotate('', xy=(0.28, 0.8), xytext=(0.22, 0.8), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax2.annotate('', xy=(0.53, 0.8), xytext=(0.47, 0.8), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax2.annotate('', xy=(0.78, 0.8), xytext=(0.72, 0.8), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# Labels
ax2.text(0.25, 0.7, 'git push', fontsize=8, ha='center')
ax2.text(0.5, 0.7, 'webhook', fontsize=8, ha='center')
ax2.text(0.75, 0.7, 'build', fontsize=8, ha='center')

# Continuous deployment
ax2.annotate('', xy=(0.4, 0.55), xytext=(0.65, 0.55),
            arrowprops=dict(arrowstyle='<->', lw=2, color=MLGREEN))
ax2.text(0.525, 0.45, 'Auto-deploy on push!', fontsize=10, ha='center', color=MLGREEN, fontweight='bold')

# URL example
ax2.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.12, facecolor=MLLAVENDER, alpha=0.5))
ax2.text(0.5, 0.26, 'Your App URL:', fontsize=10, ha='center', fontweight='bold')
ax2.text(0.5, 0.22, 'https://username-repo-main-app.streamlit.app', fontsize=8, ha='center', fontfamily='monospace')

ax2.set_xlim(0, 1)
ax2.set_ylim(0.1, 1)
ax2.set_title('Deployment Workflow', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Common issues
ax3 = axes[1, 0]
ax3.axis('off')

issues = '''
COMMON DEPLOYMENT ISSUES

1. MISSING DEPENDENCIES:
------------------------
Error: ModuleNotFoundError
Fix: Add package to requirements.txt


2. REQUIREMENTS.TXT ISSUES:
---------------------------
Error: Could not find version
Fix: Check package names and versions

# Bad
sklearn==1.0.0

# Good
scikit-learn==1.0.0


3. FILE NOT FOUND:
------------------
Error: FileNotFoundError
Fix: Use relative paths from app.py location

# Bad
df = pd.read_csv("C:/Users/data.csv")

# Good
df = pd.read_csv("data/stocks.csv")


4. MEMORY EXCEEDED:
-------------------
Error: Killed (out of memory)
Fix:
- Reduce data size
- Use caching efficiently
- Upgrade plan


5. SECRETS NOT FOUND:
---------------------
Error: st.secrets key error
Fix: Add secrets in Streamlit Cloud UI
     (Settings -> Secrets)


6. PYTHON VERSION:
------------------
Error: Syntax error (version mismatch)
Fix: Add .python-version file
     or specify in runtime.txt

# .python-version
3.10
'''

ax3.text(0.02, 0.98, issues, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Common Issues', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Success checklist
ax4 = axes[1, 1]

checklist = [
    ('requirements.txt exists', True),
    ('All imports in requirements', True),
    ('No absolute file paths', True),
    ('Secrets configured in cloud', True),
    ('Data files in repository', True),
    ('Model files in repository', True),
    ('Tested locally first', True),
    ('README.md included', True)
]

y_pos = np.arange(len(checklist))

for i, (item, status) in enumerate(checklist):
    color = MLGREEN if status else MLRED
    ax4.barh(i, 1, color=color, alpha=0.3, edgecolor='black')
    symbol = '[OK]' if status else '[  ]'
    ax4.text(0.05, i, f'{symbol} {item}', fontsize=10, va='center')

ax4.set_xlim(0, 1.1)
ax4.set_ylim(-0.5, len(checklist)-0.5)
ax4.invert_yaxis()
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_title('Pre-Deployment Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
