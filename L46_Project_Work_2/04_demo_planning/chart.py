"""Demo Planning - Preparing Your Live Demonstration"""
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
fig.suptitle('Demo Planning: Preparing Your Live Demonstration', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Demo script
ax1 = axes[0, 0]
ax1.axis('off')

demo_script = '''
DEMO SCRIPT (90 SECONDS)

1. INTRODUCTION (10 sec)
------------------------
"Let me show you the app in action."
Open app in browser (pre-loaded!)


2. INTERFACE OVERVIEW (15 sec)
------------------------------
"On the left, you see the input panel.
The main area shows our analysis."
Point to key UI elements.


3. INPUT EXAMPLE (20 sec)
-------------------------
"Let's select Apple stock..."
Make ONE selection.
"And choose a prediction timeframe..."
Set parameters.


4. SHOW PREDICTION (25 sec)
---------------------------
"When I click Predict..."
Click button.
"The model shows [result]."
Explain what the result means.


5. VISUALIZATION (15 sec)
-------------------------
"This chart shows..."
Point to key insights.
"Notice how [observation]."


6. WRAP UP (5 sec)
------------------
"That's the app in action."
Return to slides.


TOTAL: ~90 seconds

PRE-DEMO CHECKLIST:
-------------------
[ ] App URL bookmarked
[ ] Browser tab already open
[ ] Data pre-loaded
[ ] Know exact clicks needed
'''

ax1.text(0.02, 0.98, demo_script, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Demo Script', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: What can go wrong
ax2 = axes[0, 1]

issues = [
    'App not loading',
    'Slow response',
    'Wrong prediction',
    'UI breaks',
    'Network issues',
    'Screen sharing fails'
]
likelihood = [15, 30, 10, 20, 25, 40]
impact = [90, 40, 30, 50, 80, 70]

ax2.scatter(likelihood, impact, s=200, c=[MLRED, MLORANGE, MLGREEN, MLORANGE, MLRED, MLRED], alpha=0.7)

for i, issue in enumerate(issues):
    ax2.annotate(issue, (likelihood[i], impact[i]), textcoords='offset points',
                xytext=(5, 5), fontsize=8)

ax2.set_xlabel('Likelihood (%)')
ax2.set_ylabel('Impact (severity)')
ax2.set_xlim(0, 50)
ax2.set_ylim(0, 100)
ax2.set_title('Risk Matrix: What Can Go Wrong', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3)

# Add quadrant labels
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax2.axvline(x=25, color='gray', linestyle='--', alpha=0.3)
ax2.text(12, 75, 'High Impact\nLow Likelihood', fontsize=7, ha='center', color='gray')
ax2.text(37, 75, 'High Impact\nHigh Likelihood', fontsize=7, ha='center', color='gray')

# Plot 3: Backup plans
ax3 = axes[1, 0]
ax3.axis('off')

backup = '''
BACKUP PLANS

IF APP DOESN'T LOAD:
--------------------
Plan A: Refresh page, wait 30 sec
Plan B: Have LOCAL backup ready!
        "streamlit run app.py"
Plan C: Show screenshots in slides


IF APP IS SLOW:
---------------
- Pre-load data before demo
- Have results cached
- Talk while waiting
- "The model is analyzing..."


IF PREDICTION IS WRONG:
-----------------------
- Stay calm! This is ML.
- "Interesting, let's see why..."
- Have a working example ready
- Explain model limitations


IF NETWORK FAILS:
-----------------
- Run LOCAL version
- Have screenshots as backup
- Pre-record a video backup


GENERAL BACKUP STRATEGY:
------------------------
1. Always have local version ready
2. Pre-record demo video (optional)
3. Include key screenshots in slides
4. Know your app's failure modes
5. Practice recovery scenarios


WHAT TO SAY IF THINGS FAIL:
---------------------------
"Let me switch to my backup..."
"As you can see from this screenshot..."
"I'll show the local version instead..."

NEVER say:
"I don't know why it's not working..."
"This worked before..."
'''

ax3.text(0.02, 0.98, backup, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Backup Plans', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Demo flow diagram
ax4 = axes[1, 1]
ax4.axis('off')

# Draw demo flow
ax4.text(0.5, 0.95, 'DEMO FLOW', fontsize=12, ha='center', fontweight='bold')

# Main flow
steps = [
    ('OPEN', 'Pre-loaded\napp', 0.1, 0.7, MLBLUE),
    ('INPUT', 'Select\noptions', 0.3, 0.7, MLORANGE),
    ('PREDICT', 'Click\nbutton', 0.5, 0.7, MLGREEN),
    ('SHOW', 'Explain\nresults', 0.7, 0.7, MLPURPLE),
    ('CLOSE', 'Return to\nslides', 0.9, 0.7, MLBLUE)
]

for label, desc, x, y, color in steps:
    ax4.add_patch(plt.Circle((x, y), 0.07, facecolor=color, alpha=0.3))
    ax4.text(x, y+0.01, label, fontsize=9, ha='center', va='center', fontweight='bold')
    ax4.text(x, y-0.15, desc, fontsize=7, ha='center')

# Arrows
for i in range(len(steps)-1):
    x1 = steps[i][2] + 0.07
    x2 = steps[i+1][2] - 0.07
    ax4.annotate('', xy=(x2, 0.7), xytext=(x1, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

# Timing bar
ax4.add_patch(plt.Rectangle((0.1, 0.4), 0.8, 0.08, facecolor=MLLAVENDER))
segments = [(0.1, 0.11, '10s'), (0.21, 0.16, '15s'), (0.37, 0.21, '20s'),
            (0.58, 0.27, '25s'), (0.85, 0.05, '5s+')]
colors = [MLBLUE, MLORANGE, MLGREEN, MLPURPLE, MLBLUE]
start = 0.1
for i, (s, w, t) in enumerate(segments):
    ax4.add_patch(plt.Rectangle((start, 0.4), w, 0.08, facecolor=colors[i], alpha=0.5))
    ax4.text(start + w/2, 0.44, t, fontsize=7, ha='center', va='center')
    start += w

ax4.text(0.5, 0.35, 'Timeline (90 seconds total)', fontsize=8, ha='center')

# Tips
ax4.text(0.5, 0.22, 'PRO TIPS:', fontsize=10, ha='center', fontweight='bold', color=MLGREEN)
tips = ['- Practice 3+ times before presentation',
        '- Know EXACTLY what to click',
        '- Talk slowly, demo quickly']
for j, tip in enumerate(tips):
    ax4.text(0.5, 0.15 - j*0.05, tip, fontsize=8, ha='center')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Demo Flow', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
