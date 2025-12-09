"""Time Management - Staying Within Your Time Limit"""
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
fig.suptitle('Time Management: Staying Within Your Time Limit', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Timing breakdown
ax1 = axes[0, 0]
ax1.axis('off')

timing = '''
TIMING YOUR PRESENTATION

TARGET: 5-7 MINUTES
--------------------
Minimum: 5 minutes (too short = rushed)
Ideal: 6-7 minutes (just right)
Maximum: 7 minutes (will be stopped!)


SECTION TIMING:
---------------
Opening:        30 sec (0:00 - 0:30)
Problem:        60 sec (0:30 - 1:30)
Data/Approach: 90 sec (1:30 - 3:00)
Results:        60 sec (3:00 - 4:00)
Demo:          120 sec (4:00 - 6:00)
Conclusion:     60 sec (6:00 - 7:00)


TIMING TIPS:
------------
1. Practice with a timer
2. Note checkpoint times
3. Have "skip" sections ready
4. Know your pace (words/min)
5. Build in buffer time


AVERAGE SPEAKING PACE:
----------------------
Normal: 120-150 words/minute
Presentations: 100-120 words/minute
Slow/clear: 80-100 words/minute

For 7 minutes = ~700-800 words


SLIDE TIMING:
-------------
Rule: 1-2 minutes per content slide
With 7 slides = about right for 7 min


PRACTICE SCHEDULE:
------------------
Day -3: First full run (expect 10+ min)
Day -2: Cut content, aim for 8 min
Day -1: Polish, aim for 7 min
Day 0: Final practice, nail 7 min!
'''

ax1.text(0.02, 0.98, timing, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Timing Breakdown', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Time checkpoints
ax2 = axes[0, 1]

checkpoints = [
    (0, 'Start'),
    (0.5, 'Problem'),
    (1.5, 'Data'),
    (3, 'Results'),
    (4, 'Demo Start'),
    (6, 'Conclusion'),
    (7, 'End')
]

times = [c[0] for c in checkpoints]
labels = [c[1] for c in checkpoints]

ax2.plot(times, [1]*len(times), 'o-', markersize=15, color=MLBLUE, linewidth=3)

for t, label in checkpoints:
    ax2.annotate(label, (t, 1), textcoords='offset points',
                xytext=(0, 20), ha='center', fontsize=9,
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
    ax2.annotate(f'{t} min', (t, 1), textcoords='offset points',
                xytext=(0, -25), ha='center', fontsize=8)

# Danger zones
ax2.axvspan(6.5, 7.5, alpha=0.2, color=MLRED, label='Danger zone')
ax2.axvspan(0, 4.5, alpha=0.1, color=MLGREEN, label='Safe zone')

ax2.set_xlim(-0.5, 7.5)
ax2.set_ylim(0.5, 1.5)
ax2.set_xlabel('Time (minutes)')
ax2.set_yticks([])
ax2.legend(fontsize=8, loc='upper right')
ax2.set_title('Time Checkpoints', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: What to cut if running long
ax3 = axes[1, 0]
ax3.axis('off')

cut_options = '''
IF RUNNING LONG - WHAT TO CUT

PRIORITY 1 (NEVER CUT):
-----------------------
- Problem statement
- Key results
- Live demo
- Conclusion


PRIORITY 2 (CUT LAST):
----------------------
- Model comparison details
- Data preprocessing steps
- Main visualizations


PRIORITY 3 (CUT FIRST):
-----------------------
- Additional EDA charts
- Detailed methodology
- Extra features demo
- Minor results
- Extensive background


SKIP SECTIONS:
--------------
Prepare "express" versions:
- 30-second data section
- 60-second results summary
- Quick demo (1 feature only)


VERBAL SHORTCUTS:
-----------------
Instead of explaining details:
"As you can see in the chart..."
"I'll skip the details, but..."
"The full methodology is in my repo..."


EMERGENCY PROTOCOL:
-------------------
If at 5:00 and not at demo:
1. Say "Let me jump to the demo"
2. Skip remaining slides
3. Show one demo workflow
4. Quick conclusion
5. Apologize for rushing

Better to show working demo
than explain all theory!


WARNING SIGNS:
--------------
- 2:00 and still on problem
- 4:00 and not started results
- 6:00 and no demo yet

ADJUST IMMEDIATELY!
'''

ax3.text(0.02, 0.98, cut_options, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('What to Cut', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Practice schedule
ax4 = axes[1, 1]

days = ['Day -3', 'Day -2', 'Day -1', 'Day 0']
duration = [10, 8, 7.5, 7]
confidence = [2, 3, 4, 5]

x = np.arange(len(days))
width = 0.35

ax4_twin = ax4.twinx()

bars = ax4.bar(x - width/2, duration, width, label='Duration (min)', color=MLBLUE, alpha=0.7)
line = ax4_twin.plot(x, confidence, 'o-', color=MLGREEN, linewidth=2, markersize=10, label='Confidence')

ax4.set_ylabel('Duration (minutes)', color=MLBLUE)
ax4_twin.set_ylabel('Confidence (1-5)', color=MLGREEN)
ax4.set_xticks(x)
ax4.set_xticklabels(days)
ax4.axhline(y=7, color=MLRED, linestyle='--', alpha=0.5, label='Target')
ax4.set_ylim(0, 12)
ax4_twin.set_ylim(0, 6)
ax4.legend(loc='upper left', fontsize=8)
ax4_twin.legend(loc='upper right', fontsize=8)
ax4.set_title('Practice Progress', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
