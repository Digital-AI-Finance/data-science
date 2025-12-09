"""Cloud Deployment Options - Where to Deploy"""
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
fig.suptitle('Cloud Deployment Options: Where to Deploy Your App', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Cloud options overview
ax1 = axes[0, 0]
ax1.axis('off')

options = '''
CLOUD DEPLOYMENT OPTIONS

1. STREAMLIT CLOUD (RECOMMENDED FOR STUDENTS):
----------------------------------------------
- FREE tier available
- Built for Streamlit apps
- GitHub integration
- Easy setup (minutes!)
- Custom domains (paid)

Best for: Demos, portfolios, learning


2. HEROKU:
----------
- Free tier (limited hours)
- Supports many frameworks
- More configuration needed
- Git-based deployment

Best for: Small production apps


3. AWS / GCP / AZURE:
---------------------
- Enterprise-grade
- Pay-per-use pricing
- Full control
- Complex setup

Best for: Production at scale


4. RENDER:
----------
- Free tier available
- Simple deployment
- Auto-scaling
- Good for APIs

Best for: APIs and web apps


5. HUGGING FACE SPACES:
-----------------------
- Free for ML apps
- GPU available (paid)
- Great for AI demos
- Gradio or Streamlit

Best for: ML model demos


RECOMMENDATION FOR THIS COURSE:
-------------------------------
Use Streamlit Cloud!
- Free
- Fast setup
- Perfect for project demos
'''

ax1.text(0.02, 0.98, options, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Cloud Options Overview', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Comparison table
ax2 = axes[0, 1]

platforms = ['Streamlit\nCloud', 'Heroku', 'Render', 'HF Spaces', 'AWS']
ease = [5, 3, 4, 4, 2]
cost = [5, 4, 4, 5, 2]  # 5 = cheapest
features = [3, 4, 4, 3, 5]
scalability = [3, 3, 4, 3, 5]

x = np.arange(len(platforms))
width = 0.2

bars1 = ax2.bar(x - 1.5*width, ease, width, label='Ease of Use', color=MLBLUE, edgecolor='black')
bars2 = ax2.bar(x - 0.5*width, cost, width, label='Cost (5=Free)', color=MLGREEN, edgecolor='black')
bars3 = ax2.bar(x + 0.5*width, features, width, label='Features', color=MLORANGE, edgecolor='black')
bars4 = ax2.bar(x + 1.5*width, scalability, width, label='Scalability', color=MLPURPLE, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(platforms, fontsize=9)
ax2.set_ylabel('Score (1-5)')
ax2.set_title('Platform Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=7, loc='lower right')
ax2.set_ylim(0, 6)
ax2.grid(alpha=0.3, axis='y')

# Highlight Streamlit Cloud
ax2.annotate('Best for\nStudents!', xy=(0, 5.3), fontsize=8, ha='center', color=MLGREEN)

# Plot 3: Streamlit Cloud details
ax3 = axes[1, 0]
ax3.axis('off')

streamlit_cloud = '''
STREAMLIT CLOUD DETAILS

FREE TIER INCLUDES:
-------------------
- Unlimited public apps
- 1 private app
- GitHub integration
- Custom subdomain
- 1 GB RAM
- Shared CPU

URL: share.streamlit.io


SETUP STEPS:
------------
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect GitHub account
4. Select repository
5. Click Deploy!

That's it! 5 steps, 5 minutes.


REQUIREMENTS:
-------------
- GitHub account
- Public repository (for free tier)
- requirements.txt
- Main app file (app.py or streamlit_app.py)


YOUR APP URL:
-------------
https://[username]-[repo]-[branch]-[file].streamlit.app

Example:
https://john-stock-dashboard-main-app.streamlit.app


LIMITATIONS:
------------
- 1 GB memory
- Sleeps after inactivity
- No persistent storage
- No custom domain (free tier)


GOOD FOR:
---------
- Course projects
- Demos and portfolios
- Prototypes
- Learning deployment
'''

ax3.text(0.02, 0.98, streamlit_cloud, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Streamlit Cloud Details', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Decision flowchart
ax4 = axes[1, 1]
ax4.axis('off')

# Draw flowchart
ax4.text(0.5, 0.95, 'CHOOSING A PLATFORM', fontsize=11, ha='center', fontweight='bold')

# Start
ax4.add_patch(plt.Circle((0.5, 0.8), 0.08, facecolor=MLBLUE, alpha=0.3))
ax4.text(0.5, 0.8, 'Start', fontsize=9, ha='center')

# First question
ax4.add_patch(plt.Rectangle((0.25, 0.55), 0.5, 0.12, facecolor=MLORANGE, alpha=0.3))
ax4.text(0.5, 0.61, 'Is it a Streamlit app?', fontsize=9, ha='center')

# Arrow down
ax4.annotate('', xy=(0.5, 0.67), xytext=(0.5, 0.72), arrowprops=dict(arrowstyle='->', color='gray'))

# Yes path
ax4.text(0.8, 0.61, 'Yes', fontsize=8, color=MLGREEN)
ax4.annotate('', xy=(0.9, 0.61), xytext=(0.75, 0.61), arrowprops=dict(arrowstyle='->', color=MLGREEN))
ax4.add_patch(plt.Rectangle((0.75, 0.35), 0.22, 0.12, facecolor=MLGREEN, alpha=0.5))
ax4.text(0.86, 0.41, 'Streamlit\nCloud', fontsize=9, ha='center', fontweight='bold')

# No path - Need scale?
ax4.text(0.2, 0.61, 'No', fontsize=8, color=MLRED)
ax4.annotate('', xy=(0.1, 0.5), xytext=(0.25, 0.55), arrowprops=dict(arrowstyle='->', color=MLRED))

ax4.add_patch(plt.Rectangle((0.02, 0.35), 0.3, 0.12, facecolor=MLORANGE, alpha=0.3))
ax4.text(0.17, 0.41, 'Need scale?', fontsize=9, ha='center')

# Scale answers
ax4.text(0.35, 0.41, 'Yes', fontsize=8, color=MLGREEN)
ax4.add_patch(plt.Rectangle((0.35, 0.15), 0.25, 0.1, facecolor=MLPURPLE, alpha=0.3))
ax4.text(0.475, 0.2, 'AWS/GCP', fontsize=9, ha='center')

ax4.text(0.02, 0.3, 'No', fontsize=8, color=MLBLUE)
ax4.add_patch(plt.Rectangle((0.02, 0.15), 0.25, 0.1, facecolor=MLBLUE, alpha=0.3))
ax4.text(0.145, 0.2, 'Render/Heroku', fontsize=9, ha='center')

ax4.set_xlim(0, 1)
ax4.set_ylim(0.1, 1)
ax4.set_title('Decision Guide', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
