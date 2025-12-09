"""Checkpoint 2 - Final Progress Review Before Presentation"""
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
fig.suptitle('Checkpoint 2: Final Progress Review', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Checkpoint 2 requirements
ax1 = axes[0, 0]
ax1.axis('off')

requirements = '''
CHECKPOINT 2 REQUIREMENTS

DUE: End of Day 9 (Before Final Day)

MUST BE COMPLETE:
-----------------
[ ] All 3+ models trained
[ ] Best model selected
[ ] Streamlit app working
[ ] DEPLOYED to Streamlit Cloud
[ ] App URL shared with instructor


SUBMISSION FORMAT:
------------------
Email or Slack message with:
1. Deployed app URL
2. GitHub repository URL
3. Brief status update
4. Any remaining concerns


EXAMPLE SUBMISSION:
-------------------
Subject: Checkpoint 2 - Stock Volatility

App URL:
https://john-stock-app.streamlit.app

GitHub:
https://github.com/john/stock-app

Status:
- 3 models trained (LR, RF, MLP)
- Best: Random Forest (85% accuracy)
- App deployed and working
- Presentation slides 80% done

Concerns:
- Demo takes 10 seconds to load
- Should I add more visualizations?


WHAT HAPPENS IF NOT COMPLETE:
-----------------------------
- Meet with instructor ASAP
- May need to simplify scope
- Prioritize working demo
- Still present, but note issues
'''

ax1.text(0.02, 0.98, requirements, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Checkpoint 2 Requirements', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Readiness assessment
ax2 = axes[0, 1]

categories = ['Models\nComplete', 'App\nWorking', 'Deployed', 'Slides\nReady', 'Demo\nPracticed']
ready = [100, 100, 100, 90, 70]
target = [100, 100, 100, 100, 100]

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, target, width, label='Target', color=MLGREEN, alpha=0.3)
bars2 = ax2.bar(x + width/2, ready, width, label='Typical Day 9', color=MLBLUE, alpha=0.7)

ax2.set_ylabel('Completion (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=9)
ax2.legend(fontsize=8)
ax2.set_ylim(0, 110)
ax2.set_title('Readiness Assessment', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3, axis='y')

# Add labels
for bar, val in zip(bars2, ready):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
             ha='center', fontsize=8)

# Plot 3: Day 10 schedule
ax3 = axes[1, 0]
ax3.axis('off')

schedule = '''
FINAL DAY SCHEDULE (Day 10)

BEFORE CLASS:
-------------
- Test app one more time
- Have slides open
- Have app loaded in browser
- Arrive early!


PRESENTATION ORDER:
-------------------
- Random or alphabetical
- ~10 students per hour
- 7-9 minutes each


YOUR PRESENTATION:
------------------
00:00 - Start, Title slide
01:00 - Problem & motivation
02:00 - Data description
03:00 - Approach & models
04:00 - Results & comparison
05:00 - LIVE DEMO
06:30 - Conclusion
07:00 - Q&A starts


DURING OTHERS' PRESENTATIONS:
-----------------------------
- Be respectful (no phones)
- Take notes for feedback
- Think of questions
- Learn from peers


AFTER YOUR PRESENTATION:
------------------------
- Answer Q&A confidently
- Submit final materials
- Celebrate! You're done!


GRADING HAPPENS:
----------------
During/after presentations.
Grades typically within 1 week.
'''

ax3.text(0.02, 0.98, schedule, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Final Day Schedule', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Final checklist countdown
ax4 = axes[1, 1]
ax4.axis('off')

# Draw countdown checklist
ax4.text(0.5, 0.95, 'FINAL COUNTDOWN CHECKLIST', fontsize=12, ha='center', fontweight='bold')

checklist = [
    ('DAY 9 (NOW)', [
        'App deployed and URL shared',
        'GitHub repo is public',
        'Slides 90%+ complete'
    ], MLORANGE),
    ('DAY 9 EVENING', [
        'Final testing of deployed app',
        'Slides finalized',
        'Demo practiced 2-3 times'
    ], MLBLUE),
    ('DAY 10 MORNING', [
        'Test app works',
        'Backup plan ready',
        'Arrive 15 min early'
    ], MLGREEN),
    ('PRESENTATION', [
        'Stay calm, breathe',
        'Speak clearly',
        'Show enthusiasm!'
    ], MLPURPLE)
]

y = 0.85
for section, items, color in checklist:
    ax4.add_patch(plt.Rectangle((0.02, y-0.18), 0.96, 0.2, facecolor=color, alpha=0.1))
    ax4.text(0.05, y-0.02, section, fontsize=10, fontweight='bold', color=color)
    for i, item in enumerate(items):
        ax4.text(0.1, y-0.07-i*0.04, f'[ ] {item}', fontsize=8)
    y -= 0.23

ax4.text(0.5, 0.05, 'YOU ARE READY! Good luck!', fontsize=11, ha='center',
         fontweight='bold', color=MLGREEN)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Countdown Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
