"""Deployment Workflow - Complete Process"""
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
fig.suptitle('Deployment Workflow: Complete Process', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Complete deployment checklist
ax1 = axes[0, 0]
ax1.axis('off')

checklist = '''
COMPLETE DEPLOYMENT CHECKLIST

PHASE 1: DEVELOPMENT
--------------------
[ ] App runs locally without errors
[ ] All features working
[ ] Code is clean and commented
[ ] No print() debugging statements


PHASE 2: PREPARE FILES
----------------------
[ ] requirements.txt complete
[ ] All versions pinned (==)
[ ] .gitignore includes secrets
[ ] README.md with instructions
[ ] Data files included (or loaded from URL)
[ ] Model files included


PHASE 3: TEST LOCALLY
---------------------
[ ] Fresh virtualenv test
[ ] pip install -r requirements.txt
[ ] streamlit run app.py
[ ] All features still work


PHASE 4: GITHUB
---------------
[ ] Repository created
[ ] All files committed
[ ] Pushed to main branch
[ ] No secrets in repository!


PHASE 5: DEPLOY
---------------
[ ] Connected to Streamlit Cloud
[ ] Selected correct repo/branch/file
[ ] Secrets configured (if needed)
[ ] Click Deploy!


PHASE 6: VERIFY
---------------
[ ] App loads successfully
[ ] All features work
[ ] Share URL with others
[ ] Test on mobile device
'''

ax1.text(0.02, 0.98, checklist, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Deployment Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Troubleshooting
ax2 = axes[0, 1]
ax2.axis('off')

troubleshooting = '''
DEPLOYMENT TROUBLESHOOTING

ERROR: ModuleNotFoundError
--------------------------
Cause: Package missing from requirements.txt
Fix:
1. Check package name (sklearn != scikit-learn)
2. Add to requirements.txt
3. Commit and push


ERROR: FileNotFoundError
------------------------
Cause: File path wrong or file missing
Fix:
1. Use relative paths from app.py
2. Include file in repository
3. Check .gitignore not excluding it


ERROR: MemoryError / Killed
---------------------------
Cause: App using too much memory
Fix:
1. Reduce data size
2. Use @st.cache_data
3. Load data from URL instead
4. Upgrade Streamlit Cloud plan


ERROR: Timeout during build
---------------------------
Cause: Too many/large dependencies
Fix:
1. Remove unused packages
2. Use lighter alternatives
3. Pre-compile models


ERROR: App not updating
-----------------------
Cause: Cache or deployment issue
Fix:
1. Clear browser cache
2. Reboot app in Streamlit Cloud
3. Check latest commit pushed


ERROR: Secrets not found
------------------------
Cause: Secrets not configured in cloud
Fix:
1. Go to app Settings
2. Add secrets in TOML format
3. Save and reboot


LOGS:
-----
Check logs in Streamlit Cloud:
App menu -> "Manage app" -> "Logs"
'''

ax2.text(0.02, 0.98, troubleshooting, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Troubleshooting', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Workflow visualization
ax3 = axes[1, 0]
ax3.axis('off')

# Draw workflow phases
phases = [
    ('DEVELOP', MLBLUE, ['Code app', 'Test locally', 'Fix bugs']),
    ('PREPARE', MLORANGE, ['requirements.txt', '.gitignore', 'README.md']),
    ('DEPLOY', MLGREEN, ['Push to GitHub', 'Connect Cloud', 'Configure']),
    ('MAINTAIN', MLPURPLE, ['Monitor', 'Update', 'Fix issues'])
]

for i, (name, color, steps) in enumerate(phases):
    x = 0.125 + i * 0.25
    ax3.add_patch(plt.Rectangle((x-0.1, 0.3), 0.2, 0.6, facecolor=color, alpha=0.3))
    ax3.text(x, 0.85, name, fontsize=10, ha='center', fontweight='bold')
    for j, step in enumerate(steps):
        ax3.text(x, 0.7 - j*0.12, f'- {step}', fontsize=8, ha='center')

# Arrows
for i in range(3):
    x1 = 0.225 + i * 0.25
    ax3.annotate('', xy=(x1+0.05, 0.6), xytext=(x1-0.05, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# Feedback loop
ax3.annotate('', xy=(0.125, 0.2), xytext=(0.875, 0.2),
            arrowprops=dict(arrowstyle='<-', lw=1, color='gray',
                          connectionstyle='arc3,rad=-0.3'))
ax3.text(0.5, 0.12, 'Continuous Improvement Cycle', fontsize=9, ha='center', style='italic')

ax3.set_xlim(0, 1)
ax3.set_ylim(0.05, 1)
ax3.set_title('Workflow Phases', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Timeline
ax4 = axes[1, 1]

# Create timeline
milestones = [
    ('Start', 0, 'Begin development'),
    ('Local\nComplete', 2, 'App works locally'),
    ('Files\nReady', 3, 'All files prepared'),
    ('Pushed', 3.5, 'On GitHub'),
    ('Deployed!', 4, 'Live on cloud'),
    ('Shared', 4.5, 'URL shared')
]

times = [m[1] for m in milestones]
labels = [m[0] for m in milestones]

ax4.plot(times, [0.5]*len(times), 'o-', markersize=15, color=MLBLUE)

for t, label, desc in milestones:
    ax4.annotate(f'{label}', xy=(t[1], 0.5), xytext=(t[1], 0.7),
                fontsize=9, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='-', color='gray'))
    ax4.text(t[1], 0.3, desc, fontsize=7, ha='center', style='italic')

ax4.set_xlim(-0.5, 5)
ax4.set_ylim(0, 1)
ax4.set_xlabel('Time (hours for simple app)')
ax4.set_yticks([])
ax4.set_title('Deployment Timeline', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
