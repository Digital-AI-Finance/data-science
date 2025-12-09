"""Checkpoint 1 - First Progress Review"""
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
fig.suptitle('Checkpoint 1: First Progress Review', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Checkpoint requirements
ax1 = axes[0, 0]
ax1.axis('off')

checkpoint_req = '''
CHECKPOINT 1 REQUIREMENTS

DUE: End of Day 5

WHAT TO SUBMIT:
---------------
1. Project topic (1 sentence)
2. Data source and description
3. Initial EDA results
4. First model results
5. Any blockers/questions


SUBMISSION FORMAT:
------------------
Share via email or Slack:
- Topic: [Your topic]
- Data: [Source, # rows, # features]
- Models tried: [List]
- Best result so far: [Metric]
- Questions: [Any blockers?]


EXAMPLE SUBMISSION:
-------------------
Topic: Stock volatility prediction for
       tech sector using VIX and prices

Data: Yahoo Finance, AAPL/MSFT/GOOGL
      2020-2024, 5000 rows, 15 features

Models: Linear Regression (baseline)
        R2 = 0.42, MAE = 0.15

Best: Linear Regression so far,
      trying Random Forest next

Questions: Should I add more stocks
           or focus on fewer?


THIS IS NOT GRADED:
-------------------
Purpose is to ensure you're on track.
No points assigned to checkpoint.
But MUST complete to continue.
'''

ax1.text(0.02, 0.98, checkpoint_req, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Checkpoint Requirements', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Progress status visualization
ax2 = axes[0, 1]

categories = ['Topic\nSelected', 'Data\nAcquired', 'EDA\nDone', 'Model\nStarted', 'On\nSchedule']
target = [100, 100, 100, 50, 100]  # Target for checkpoint 1
example_good = [100, 100, 80, 60, 90]  # Good progress
example_behind = [100, 50, 20, 0, 40]  # Behind schedule

x = np.arange(len(categories))
width = 0.25

bars1 = ax2.bar(x - width, target, width, label='Target', color=MLGREEN, alpha=0.3)
bars2 = ax2.bar(x, example_good, width, label='Good Progress', color=MLGREEN, alpha=0.7)
bars3 = ax2.bar(x + width, example_behind, width, label='Behind Schedule', color=MLRED, alpha=0.7)

ax2.set_ylabel('Completion (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=9)
ax2.legend(fontsize=8)
ax2.set_ylim(0, 110)
ax2.set_title('Progress Status Examples', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3, axis='y')

# Plot 3: Common issues and solutions
ax3 = axes[1, 0]
ax3.axis('off')

issues = '''
COMMON CHECKPOINT 1 ISSUES

ISSUE: Can't find good data
SOLUTION:
- Use Yahoo Finance (always works)
- Try Kaggle datasets
- Generate synthetic data
- Ask instructor for help

ISSUE: Data has many missing values
SOLUTION:
- Document % missing
- Use dropna() or fillna()
- Explain your approach
- This is expected!

ISSUE: Model results are poor
SOLUTION:
- This is NORMAL at checkpoint
- Document baseline results
- Will improve with tuning
- Focus on pipeline, not metrics yet

ISSUE: Topic seems too hard
SOLUTION:
- Simplify scope
- Focus on one prediction task
- Better to finish simple project
- Ask for scope adjustment

ISSUE: Running behind schedule
SOLUTION:
- Get help immediately!
- Focus on MVP (minimum viable)
- Skip nice-to-haves
- Work on essentials first


REMEMBER:
---------
Checkpoint is for feedback.
Better to show problems early
than hide them until final day!
'''

ax3.text(0.02, 0.98, issues, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Common Issues & Solutions', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Next steps after checkpoint
ax4 = axes[1, 1]
ax4.axis('off')

# Draw next steps flow
ax4.text(0.5, 0.95, 'AFTER CHECKPOINT 1', fontsize=12, ha='center', fontweight='bold')

steps = [
    ('RECEIVE\nFEEDBACK', 'Review instructor\ncomments', MLBLUE, 0.15),
    ('ADJUST\nPLAN', 'Modify scope\nif needed', MLORANGE, 0.38),
    ('CONTINUE\nDEVELOPMENT', 'More models,\nstart app', MLGREEN, 0.62),
    ('PREPARE FOR\nCHECKPOINT 2', 'Day 9:\nDeployment', MLPURPLE, 0.85)
]

for title, desc, color, x in steps:
    ax4.add_patch(plt.Rectangle((x-0.1, 0.4), 0.2, 0.45, facecolor=color, alpha=0.2))
    ax4.text(x, 0.78, title, fontsize=9, ha='center', fontweight='bold', color=color)
    ax4.text(x, 0.55, desc, fontsize=8, ha='center')

# Arrows
for i in range(len(steps)-1):
    x1 = steps[i][3] + 0.1
    x2 = steps[i+1][3] - 0.1
    ax4.annotate('', xy=(x2, 0.62), xytext=(x1, 0.62),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# Key message
ax4.add_patch(plt.Rectangle((0.1, 0.08), 0.8, 0.2, facecolor=MLGREEN, alpha=0.2))
ax4.text(0.5, 0.22, 'KEY MESSAGE:', fontsize=10, ha='center', fontweight='bold', color=MLGREEN)
ax4.text(0.5, 0.12, 'Checkpoint 1 is your chance to course-correct.\nUse the feedback to improve your final project!',
         fontsize=9, ha='center')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Next Steps', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
