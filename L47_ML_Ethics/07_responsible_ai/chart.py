"""Responsible AI - Building Ethical ML Systems"""
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
fig.suptitle('Responsible AI: Building Ethical ML Systems', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Responsible AI principles
ax1 = axes[0, 0]
ax1.axis('off')

principles = '''
RESPONSIBLE AI PRINCIPLES

1. HUMAN-CENTERED:
------------------
- AI should augment humans
- Not replace human judgment
- Enable human oversight
- Respect human autonomy


2. INCLUSIVE:
-------------
- Works for diverse users
- Accessible to all
- Considers edge cases
- Avoids discrimination


3. TRANSPARENT:
---------------
- Clear about AI use
- Explainable decisions
- Documented methodology
- Honest about limitations


4. ACCOUNTABLE:
---------------
- Clear ownership
- Error correction process
- Feedback mechanisms
- Continuous improvement


5. SAFE & SECURE:
-----------------
- Robust to attacks
- Fail-safe mechanisms
- Data protection
- Privacy preservation


6. BENEFICIAL:
--------------
- Creates positive value
- Minimizes harm
- Considers externalities
- Long-term thinking


THE RESPONSIBLE AI DEVELOPER:
-----------------------------
"I build models that help people
 make better decisions, while
 being honest about what the
 model can and cannot do."
'''

ax1.text(0.02, 0.98, principles, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Responsible AI Principles', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Responsible AI framework
ax2 = axes[0, 1]
ax2.axis('off')

# Draw framework as connected hexagons
principles_viz = [
    ('DESIGN', 'Consider ethics\nfrom start', 0.2, 0.75, MLBLUE),
    ('DATA', 'Quality and\nfairness', 0.5, 0.85, MLORANGE),
    ('DEVELOP', 'Transparent\nmethods', 0.8, 0.75, MLGREEN),
    ('DEPLOY', 'Safe and\nmonitored', 0.8, 0.45, MLPURPLE),
    ('MAINTAIN', 'Continuous\nimprovement', 0.5, 0.35, MLRED),
    ('GOVERN', 'Oversight and\naccountability', 0.2, 0.45, MLBLUE)
]

for name, desc, x, y, color in principles_viz:
    ax2.add_patch(plt.RegularPolygon((x, y), 6, 0.12, facecolor=color, alpha=0.3))
    ax2.text(x, y+0.02, name, fontsize=9, ha='center', fontweight='bold', color=color)
    ax2.text(x, y-0.08, desc, fontsize=7, ha='center')

# Connect with lines
connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
for i, j in connections:
    x1, y1 = principles_viz[i][2], principles_viz[i][3]
    x2, y2 = principles_viz[j][2], principles_viz[j][3]
    ax2.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=2)

ax2.text(0.5, 0.95, 'RESPONSIBLE AI LIFECYCLE', fontsize=11, ha='center', fontweight='bold')
ax2.text(0.5, 0.6, 'HUMAN\nOVERSIGHT', fontsize=10, ha='center', fontweight='bold', color=MLPURPLE)
ax2.set_xlim(0, 1)
ax2.set_ylim(0.2, 1)
ax2.set_title('AI Lifecycle Framework', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Practical guidelines
ax3 = axes[1, 0]
ax3.axis('off')

guidelines = '''
PRACTICAL GUIDELINES

BEFORE BUILDING:
----------------
1. Define the problem clearly
2. Consider who is affected
3. Assess potential harms
4. Plan for edge cases
5. Get diverse input


DURING DEVELOPMENT:
-------------------
1. Use quality data
2. Test for bias
3. Document everything
4. Review with others
5. Consider alternatives


BEFORE DEPLOYMENT:
------------------
1. Validate thoroughly
2. Plan monitoring
3. Prepare for failures
4. Train users
5. Get approval


AFTER DEPLOYMENT:
-----------------
1. Monitor performance
2. Collect feedback
3. Address issues quickly
4. Update regularly
5. Report outcomes


QUESTIONS TO ASK:
-----------------
- "Who might be harmed?"
- "How could this be misused?"
- "What if I'm wrong?"
- "Would I use this myself?"
- "Can I explain this decision?"


THE ETHICS TEST:
----------------
"Would I be comfortable if my
 model's decision process was
 on the front page of a newspaper?"
'''

ax3.text(0.02, 0.98, guidelines, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Practical Guidelines', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Responsible AI maturity assessment
ax4 = axes[1, 1]

categories = ['Awareness', 'Governance', 'Implementation', 'Monitoring', 'Culture']

# Example maturity levels
beginner = [2, 1, 1, 1, 2]
intermediate = [4, 3, 3, 3, 3]
advanced = [5, 5, 5, 5, 5]

# Create radar chart
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

beginner += beginner[:1]
intermediate += intermediate[:1]
advanced += advanced[:1]

ax4 = plt.subplot(2, 2, 4, projection='polar')
ax4.plot(angles, beginner, 'o-', linewidth=2, color=MLRED, label='Beginner')
ax4.fill(angles, beginner, alpha=0.1, color=MLRED)
ax4.plot(angles, intermediate, 'o-', linewidth=2, color=MLORANGE, label='Intermediate')
ax4.fill(angles, intermediate, alpha=0.1, color=MLORANGE)
ax4.plot(angles, advanced, 'o-', linewidth=2, color=MLGREEN, label='Advanced')
ax4.fill(angles, advanced, alpha=0.1, color=MLGREEN)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=8)
ax4.set_ylim(0, 5)
ax4.legend(loc='lower right', fontsize=8)
ax4.set_title('Responsible AI Maturity', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
