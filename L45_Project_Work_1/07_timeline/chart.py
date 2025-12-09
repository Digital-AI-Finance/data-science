"""Timeline - Project Schedule and Milestones"""
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
fig.suptitle('Project Timeline: Schedule and Milestones', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Two-week timeline
ax1 = axes[0, 0]

# Gantt chart style
tasks = [
    ('Topic Selection', 0, 1, MLBLUE),
    ('Data Acquisition', 0.5, 2, MLBLUE),
    ('EDA & Cleaning', 1, 3, MLORANGE),
    ('Model Development', 2, 5, MLGREEN),
    ('Model Evaluation', 4, 6, MLGREEN),
    ('App Development', 5, 8, MLPURPLE),
    ('Deployment', 7, 9, MLRED),
    ('Testing & Polish', 8, 10, MLRED),
    ('Presentation Prep', 9, 10, MLORANGE),
    ('Final Presentation', 10, 10.5, MLRED)
]

for i, (task, start, end, color) in enumerate(tasks):
    ax1.barh(i, end-start, left=start, color=color, alpha=0.7, height=0.6)
    ax1.text(start + (end-start)/2, i, task, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax1.set_yticks(range(len(tasks)))
ax1.set_yticklabels(['' for _ in tasks])
ax1.set_xlabel('Days')
ax1.set_xlim(0, 11)
ax1.set_title('Project Gantt Chart (2 Weeks)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
ax1.text(5, -0.8, 'Week 1 End', fontsize=8, ha='center')
ax1.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
ax1.text(10, -0.8, 'Week 2 End', fontsize=8, ha='center')
ax1.invert_yaxis()
ax1.grid(alpha=0.3, axis='x')

# Plot 2: Daily breakdown
ax2 = axes[0, 1]
ax2.axis('off')

daily = '''
DAILY BREAKDOWN

WEEK 1:
-------
Day 1-2: Topic & Data
  - Finalize topic
  - Acquire data
  - Initial EDA

Day 3-4: Data Processing
  - Clean data
  - Feature engineering
  - Prepare train/test

Day 5: Models (Part 1)
  - Baseline model
  - First complex model
  - Initial evaluation


WEEK 2:
-------
Day 6-7: Models (Part 2)
  - Additional models
  - Hyperparameter tuning
  - Cross-validation
  - Model selection

Day 8: App Development
  - Streamlit app
  - UI/UX design
  - Integrate model

Day 9: Deployment
  - Push to GitHub
  - Deploy to cloud
  - Test deployed app

Day 10: Polish & Present
  - Fix bugs
  - Prepare slides
  - Practice presentation
  - FINAL PRESENTATION!


KEY DATES:
----------
Day 5: Checkpoint 1 (models started)
Day 9: Deployment complete
Day 10: Final presentation
'''

ax2.text(0.02, 0.98, daily, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Daily Breakdown', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Effort distribution
ax3 = axes[1, 0]

activities = ['Data\nWork', 'Model\nDev', 'App\nDev', 'Deploy', 'Docs &\nPresent']
hours = [8, 12, 8, 4, 4]
colors = [MLBLUE, MLGREEN, MLPURPLE, MLRED, MLORANGE]

bars = ax3.bar(activities, hours, color=colors, alpha=0.7)
ax3.set_ylabel('Hours')
ax3.set_title('Recommended Time Allocation (36 hrs total)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.grid(alpha=0.3, axis='y')

for bar, hour in zip(bars, hours):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{hour}h\n({hour/36*100:.0f}%)', ha='center', fontsize=9)

# Plot 4: Milestone checklist
ax4 = axes[1, 1]
ax4.axis('off')

milestones = '''
MILESTONE CHECKLIST

CHECKPOINT 1 (Day 5):
---------------------
[ ] Topic approved
[ ] Data acquired and cleaned
[ ] At least 1 model trained
[ ] Initial results documented


CHECKPOINT 2 (Day 9):
---------------------
[ ] All models trained
[ ] Best model selected
[ ] Streamlit app working
[ ] Deployed to cloud
[ ] URL shared with instructor


FINAL (Day 10):
---------------
[ ] App fully functional
[ ] Documentation complete
[ ] Slides ready
[ ] Demo practiced
[ ] Presentation delivered!


RED FLAGS - GET HELP IF:
------------------------
- Day 3: No data yet
- Day 5: No working model
- Day 8: No app started
- Day 9: Not deployed


INSTRUCTOR OFFICE HOURS:
------------------------
Available for questions
throughout project period.
Ask early, ask often!
'''

ax4.text(0.02, 0.98, milestones, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Milestones', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
