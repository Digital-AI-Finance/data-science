"""Presentation Guidelines - Delivering Your Final Presentation"""
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
fig.suptitle('Presentation Guidelines: Delivering Your Final Presentation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Presentation structure
ax1 = axes[0, 0]
ax1.axis('off')

structure = '''
PRESENTATION STRUCTURE (7 MINUTES)

1. OPENING (30 seconds)
-----------------------
- Introduce yourself
- State your project title
- Hook the audience


2. PROBLEM (1 minute)
---------------------
- What problem are you solving?
- Why does it matter?
- Who benefits?


3. DATA & APPROACH (1.5 minutes)
--------------------------------
- Data source
- Key statistics
- Models used
- Why these choices?


4. RESULTS (1 minute)
---------------------
- Model comparison
- Best performance
- Key insights


5. LIVE DEMO (2 minutes)
------------------------
- Show working app
- One complete workflow
- Highlight features


6. CONCLUSION (1 minute)
------------------------
- Key takeaways
- Limitations
- Future work
- Thank the audience


TOTAL: 7 minutes
Q&A: 2-3 minutes additional


GOLDEN RULE:
------------
Practice until you can do it
in your sleep!
'''

ax1.text(0.02, 0.98, structure, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Presentation Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Time allocation visualization
ax2 = axes[0, 1]

sections = ['Opening', 'Problem', 'Data/Approach', 'Results', 'Demo', 'Conclusion']
times = [0.5, 1, 1.5, 1, 2, 1]
colors = [MLBLUE, MLBLUE, MLORANGE, MLGREEN, MLPURPLE, MLRED]

# Create horizontal stacked bar
left = 0
for section, time, color in zip(sections, times, colors):
    ax2.barh(0, time, left=left, color=color, alpha=0.7, height=0.5)
    ax2.text(left + time/2, 0, f'{section}\n{time}min', ha='center', va='center', fontsize=8)
    left += time

ax2.set_xlim(0, 7)
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlabel('Time (minutes)')
ax2.set_yticks([])
ax2.axvline(x=5, color=MLRED, linestyle='--', alpha=0.5)
ax2.text(5, 0.4, 'Demo\nstarts', fontsize=8, ha='center', color=MLRED)
ax2.set_title('Time Allocation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Do's and Don'ts
ax3 = axes[1, 0]
ax3.axis('off')

dos_donts = '''
DO'S AND DON'TS

DO:
---
- Practice multiple times
- Speak clearly and slowly
- Make eye contact
- Use simple language
- Show enthusiasm
- Have backup plan ready
- Test technology beforehand
- Keep to time limit
- Know your material cold
- Engage the audience


DON'T:
------
- Read from slides
- Speak too fast
- Use jargon without explaining
- Apologize for your work
- Go over time
- Panic if demo fails
- Look at the floor
- Fidget or pace excessively
- Include too much detail
- Skip the demo


IF SOMETHING GOES WRONG:
------------------------
- Stay calm
- Acknowledge briefly
- Move to backup
- Keep going

"Let me show you this
 another way..."

"I'll demonstrate using
 my local backup..."


REMEMBER:
---------
The audience wants you
to succeed!
'''

ax3.text(0.02, 0.98, dos_donts, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title("Do's and Don'ts", fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Presentation checklist
ax4 = axes[1, 1]

checklist = [
    ('Slides ready', 'All content finalized'),
    ('Demo tested', 'App works, loaded'),
    ('Backup ready', 'Screenshots, local version'),
    ('Practiced 3x', 'Timed yourself'),
    ('Notes prepared', 'Key points written'),
    ('Technology tested', 'Projector, microphone'),
    ('Arrive early', '15 minutes before'),
    ('Water bottle', 'Stay hydrated')
]

y_pos = np.arange(len(checklist))

for i, (item, desc) in enumerate(checklist):
    color = MLGREEN
    ax4.add_patch(plt.Rectangle((0, i-0.4), 0.04, 0.8, facecolor=MLLAVENDER, edgecolor='gray'))
    ax4.text(0.06, i, item, fontsize=10, va='center', fontweight='bold')
    ax4.text(0.45, i, desc, fontsize=9, va='center', color='gray')

ax4.set_xlim(0, 1)
ax4.set_ylim(-0.5, len(checklist)-0.5)
ax4.invert_yaxis()
ax4.axis('off')
ax4.set_title('Pre-Presentation Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
