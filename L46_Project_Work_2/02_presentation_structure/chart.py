"""Presentation Structure - Organizing Your Final Presentation"""
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
fig.suptitle('Presentation Structure: Organizing Your Final Presentation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Slide structure
ax1 = axes[0, 0]
ax1.axis('off')

structure = '''
PRESENTATION STRUCTURE (5-7 MIN)

SLIDE 1: TITLE (15 sec)
-----------------------
- Project title
- Your name
- Date


SLIDE 2: PROBLEM (45 sec)
-------------------------
- What problem are you solving?
- Why does it matter?
- Who would use this?


SLIDE 3: DATA (45 sec)
----------------------
- Data source
- Key statistics
- One key visualization


SLIDE 4: APPROACH (60 sec)
--------------------------
- Models tried
- Why these models?
- Key findings


SLIDE 5: RESULTS (45 sec)
-------------------------
- Best model performance
- Comparison table/chart
- Interpretation


SLIDE 6: DEMO (90 sec)
----------------------
- Live app demonstration
- Show key features
- Interactive example


SLIDE 7: CONCLUSION (30 sec)
----------------------------
- Key takeaways
- Limitations
- Future improvements


TOTAL: ~6 minutes
Q&A: 1-2 minutes
'''

ax1.text(0.02, 0.98, structure, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Slide Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Time allocation
ax2 = axes[0, 1]

sections = ['Title', 'Problem', 'Data', 'Approach', 'Results', 'Demo', 'Conclusion']
times = [15, 45, 45, 60, 45, 90, 30]
colors = [MLBLUE, MLBLUE, MLORANGE, MLGREEN, MLGREEN, MLPURPLE, MLRED]

# Convert to cumulative for stacked effect
wedges, texts, autotexts = ax2.pie(times, labels=sections, autopct=lambda p: f'{int(p*sum(times)/100)}s',
                                    colors=colors, startangle=90,
                                    wedgeprops=dict(alpha=0.7))

ax2.set_title('Time Allocation (seconds)', fontsize=11, fontweight='bold', color=MLPURPLE)

for autotext in autotexts:
    autotext.set_fontsize(8)

# Plot 3: What to include on each slide
ax3 = axes[1, 0]
ax3.axis('off')

content = '''
SLIDE CONTENT GUIDELINES

PROBLEM SLIDE:
--------------
DO:
- State problem clearly in one sentence
- Explain real-world impact
- Show who benefits

DON'T:
- Be too technical
- Use jargon
- Show code


DATA SLIDE:
-----------
DO:
- Show data source
- Include key statistics
- One compelling chart

DON'T:
- Show all your EDA
- Include raw data tables
- Overwhelm with numbers


APPROACH SLIDE:
---------------
DO:
- List models tried
- Explain why you chose them
- Mention key challenges

DON'T:
- Show code
- Go into math details
- List all hyperparameters


RESULTS SLIDE:
--------------
DO:
- Show best model clearly
- Use comparison table
- Interpret the numbers

DON'T:
- Show all experiments
- Use unexplained metrics
- Claim unrealistic accuracy


DEMO SLIDE:
-----------
DO:
- Have app open and ready
- Show one complete workflow
- Highlight key features

DON'T:
- Debug during demo
- Show everything
- Panic if something fails
'''

ax3.text(0.02, 0.98, content, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Content Guidelines', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Presentation flow diagram
ax4 = axes[1, 1]
ax4.axis('off')

# Draw flow
flow_items = [
    ('HOOK', 'Grab attention\nwith problem', 0.15, MLBLUE),
    ('CONTEXT', 'Data and\napproach', 0.38, MLORANGE),
    ('EVIDENCE', 'Results and\ndemo', 0.62, MLGREEN),
    ('CLOSE', 'Takeaways\nand future', 0.85, MLPURPLE)
]

for label, desc, x, color in flow_items:
    ax4.add_patch(plt.Circle((x, 0.7), 0.1, facecolor=color, alpha=0.3))
    ax4.text(x, 0.7, label, fontsize=10, ha='center', va='center', fontweight='bold')
    ax4.text(x, 0.48, desc, fontsize=8, ha='center')

# Arrows
for i in range(len(flow_items)-1):
    x1 = flow_items[i][2] + 0.1
    x2 = flow_items[i+1][2] - 0.1
    ax4.annotate('', xy=(x2, 0.7), xytext=(x1, 0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax4.text(0.5, 0.95, 'PRESENTATION FLOW', fontsize=12, ha='center', fontweight='bold')

# Tips
ax4.text(0.5, 0.28, 'KEY PRINCIPLES:', fontsize=10, ha='center', fontweight='bold')
principles = [
    '1. Tell a story, not a report',
    '2. One main idea per slide',
    '3. Less text, more visuals',
    '4. Practice makes perfect'
]
for j, p in enumerate(principles):
    ax4.text(0.5, 0.2 - j*0.05, p, fontsize=9, ha='center')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Presentation Flow', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
