"""GitHub Integration - Version Control for Deployment"""
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
fig.suptitle('GitHub Integration: Version Control for Deployment', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Git basics
ax1 = axes[0, 0]
ax1.axis('off')

git_basics = '''
GIT BASICS FOR DEPLOYMENT

INITIALIZE REPOSITORY:
----------------------
# In your project folder
git init


STAGE AND COMMIT:
-----------------
# Add all files
git add .

# Commit with message
git commit -m "Initial commit"


CREATE GITHUB REPO:
-------------------
1. Go to github.com
2. Click "New repository"
3. Name: stock-dashboard
4. Keep it PUBLIC (for free Streamlit Cloud)
5. Don't add README (you have one)
6. Click "Create repository"


CONNECT AND PUSH:
-----------------
# Add remote
git remote add origin https://github.com/USER/REPO.git

# Push to GitHub
git push -u origin main


DAILY WORKFLOW:
---------------
# Make changes to your code
# ...

# Check what changed
git status

# Stage changes
git add .

# Commit
git commit -m "Add new feature"

# Push to GitHub (triggers redeploy!)
git push
'''

ax1.text(0.02, 0.98, git_basics, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Git Basics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: .gitignore
ax2 = axes[0, 1]
ax2.axis('off')

gitignore = '''
.GITIGNORE - WHAT TO EXCLUDE

CREATE .gitignore FILE:
-----------------------
# .gitignore

# Secrets - CRITICAL!
.streamlit/secrets.toml
.env
*.pem
*credentials*

# Python cache
__pycache__/
*.py[cod]
*$py.class
.Python

# Virtual environments
venv/
env/
.venv/

# IDE settings
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db

# Data files (if too large)
*.csv
*.xlsx
data/raw/

# Model files (if too large)
models/*.pkl

# Jupyter checkpoints
.ipynb_checkpoints/

# Test coverage
htmlcov/
.coverage


IMPORTANT FILES TO INCLUDE:
---------------------------
- app.py (your app!)
- requirements.txt
- README.md
- Small data files
- Model files (if <100MB)
- .streamlit/config.toml


GITHUB FILE SIZE LIMITS:
------------------------
- Single file: 100 MB max
- Repository: 1 GB recommended
- For large files: Use Git LFS
'''

ax2.text(0.02, 0.98, gitignore, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('.gitignore File', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Continuous deployment flow
ax3 = axes[1, 0]
ax3.axis('off')

# Draw CD flow
ax3.text(0.5, 0.95, 'CONTINUOUS DEPLOYMENT FLOW', fontsize=11, ha='center', fontweight='bold')

# Local
ax3.add_patch(plt.Rectangle((0.05, 0.65), 0.25, 0.2, facecolor=MLBLUE, alpha=0.3))
ax3.text(0.175, 0.8, 'LOCAL', fontsize=10, ha='center', fontweight='bold')
ax3.text(0.175, 0.72, 'Edit code\nTest locally', fontsize=8, ha='center')

# Git push arrow
ax3.annotate('', xy=(0.38, 0.75), xytext=(0.32, 0.75),
            arrowprops=dict(arrowstyle='->', lw=2, color=MLORANGE))
ax3.text(0.35, 0.82, 'git push', fontsize=8, ha='center', color=MLORANGE)

# GitHub
ax3.add_patch(plt.Rectangle((0.4, 0.65), 0.2, 0.2, facecolor=MLORANGE, alpha=0.3))
ax3.text(0.5, 0.8, 'GITHUB', fontsize=10, ha='center', fontweight='bold')
ax3.text(0.5, 0.72, 'Repository\nupdated', fontsize=8, ha='center')

# Webhook arrow
ax3.annotate('', xy=(0.68, 0.75), xytext=(0.62, 0.75),
            arrowprops=dict(arrowstyle='->', lw=2, color=MLGREEN))
ax3.text(0.65, 0.82, 'webhook', fontsize=8, ha='center', color=MLGREEN)

# Streamlit Cloud
ax3.add_patch(plt.Rectangle((0.7, 0.65), 0.25, 0.2, facecolor=MLGREEN, alpha=0.3))
ax3.text(0.825, 0.8, 'STREAMLIT', fontsize=10, ha='center', fontweight='bold')
ax3.text(0.825, 0.72, 'Auto-rebuild\nAuto-deploy', fontsize=8, ha='center')

# Timeline
ax3.add_patch(plt.Rectangle((0.1, 0.3), 0.8, 0.2, facecolor=MLLAVENDER, alpha=0.3))
ax3.text(0.5, 0.45, 'TIMELINE', fontsize=10, ha='center', fontweight='bold')

timeline = [
    (0.15, '0s\nPush'),
    (0.35, '5s\nDetect'),
    (0.55, '1-3min\nBuild'),
    (0.75, '3-5min\nLive!'),
]

for x, label in timeline:
    ax3.add_patch(plt.Circle((x, 0.35), 0.04, facecolor=MLPURPLE, alpha=0.5))
    ax3.text(x, 0.35, label, fontsize=7, ha='center', va='center')

ax3.set_xlim(0, 1)
ax3.set_ylim(0.2, 1)
ax3.set_title('Deployment Flow', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Common Git commands
ax4 = axes[1, 1]

commands = [
    ('git status', 'Check what changed'),
    ('git add .', 'Stage all changes'),
    ('git commit -m "msg"', 'Save changes locally'),
    ('git push', 'Push to GitHub'),
    ('git pull', 'Get latest from GitHub'),
    ('git log --oneline', 'View commit history'),
    ('git diff', 'See changes in files'),
    ('git checkout file.py', 'Undo changes to file'),
]

# Create table
table_data = [[cmd, desc] for cmd, desc in commands]
table = ax4.table(cellText=table_data,
                  colLabels=['Command', 'Description'],
                  loc='center', cellLoc='left',
                  colColours=[MLLAVENDER, MLLAVENDER])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)

# Style command column
for i in range(len(commands)):
    table[i+1, 0].set_text_props(fontfamily='monospace', fontweight='bold')

ax4.axis('off')
ax4.set_title('Common Git Commands', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
