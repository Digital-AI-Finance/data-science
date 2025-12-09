"""Peer Evaluation - Learning from Your Classmates"""
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
fig.suptitle('Peer Evaluation: Learning from Your Classmates', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why peer evaluation
ax1 = axes[0, 0]
ax1.axis('off')

why_peer = '''
WHY PEER EVALUATION?

LEARNING OPPORTUNITY:
---------------------
- See different approaches
- Learn new techniques
- Understand what works
- Identify common mistakes


PROFESSIONAL SKILLS:
--------------------
- Critical thinking
- Constructive feedback
- Technical communication
- Collaboration


WHAT YOU'LL EVALUATE:
---------------------
- Technical implementation
- Presentation quality
- Demo effectiveness
- Documentation
- Overall impression


YOUR ROLE AS EVALUATOR:
-----------------------
1. Pay attention
2. Take notes
3. Be fair and objective
4. Give constructive feedback
5. Learn from others


YOUR ROLE AS PRESENTER:
-----------------------
1. Present your best work
2. Accept feedback gracefully
3. Learn from suggestions
4. Thank your peers


BENEFITS:
---------
- Fresh perspective on your work
- Ideas for improvement
- Recognition of strengths
- Community building


THE GOAL:
---------
Everyone learns and improves!
'''

ax1.text(0.02, 0.98, why_peer, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Why Peer Evaluation?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Evaluation criteria
ax2 = axes[0, 1]

criteria = ['Technical\nQuality', 'Presentation\nClarity', 'Demo\nEffectiveness', 'Q&A\nHandling', 'Overall\nImpression']
weights = [30, 25, 25, 10, 10]
colors = [MLGREEN, MLBLUE, MLPURPLE, MLORANGE, MLRED]

bars = ax2.bar(criteria, weights, color=colors, alpha=0.7)
ax2.set_ylabel('Weight (%)')
ax2.set_ylim(0, 40)
ax2.set_title('Peer Evaluation Criteria', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3, axis='y')

for bar, w in zip(bars, weights):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{w}%', ha='center', fontsize=10, fontweight='bold')

# Plot 3: Giving good feedback
ax3 = axes[1, 0]
ax3.axis('off')

feedback_guide = '''
GIVING CONSTRUCTIVE FEEDBACK

THE SANDWICH METHOD:
--------------------
1. Positive observation
2. Suggestion for improvement
3. Encouraging close


EXAMPLE FEEDBACK:
-----------------
"Your visualization of the model
comparison was really clear and
helped me understand the results.

One thing that might help is
adding confidence intervals to
show the uncertainty.

Overall, great project - I learned
a lot about Random Forests!"


FEEDBACK CATEGORIES:
--------------------
TECHNICAL:
- Model choice and implementation
- Data handling approach
- Evaluation methodology

PRESENTATION:
- Slide design and clarity
- Speaking pace and clarity
- Time management

DEMO:
- App functionality
- User experience
- Error handling


RATING SCALE:
-------------
5 = Exceptional (top 10%)
4 = Good (above average)
3 = Satisfactory (meets expectations)
2 = Needs improvement
1 = Significant issues


BE SPECIFIC:
------------
BAD: "Good presentation"
GOOD: "The chart on slide 4 clearly
      showed the performance difference"

BAD: "Demo was confusing"
GOOD: "Adding labels to the dropdown
      would help users understand
      the options"
'''

ax3.text(0.02, 0.98, feedback_guide, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Giving Good Feedback', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Peer evaluation form
ax4 = axes[1, 1]
ax4.axis('off')

form = '''
PEER EVALUATION FORM

Presenter: _________________________
Evaluator: _________________________


TECHNICAL QUALITY (1-5): ___
- Model implementation
- Data handling
- Results interpretation
Comments: _________________________


PRESENTATION CLARITY (1-5): ___
- Slide design
- Speaking clarity
- Time management
Comments: _________________________


DEMO EFFECTIVENESS (1-5): ___
- App functionality
- User experience
- Error handling
Comments: _________________________


Q&A HANDLING (1-5): ___
- Answer quality
- Confidence
- Honesty about limitations
Comments: _________________________


OVERALL IMPRESSION (1-5): ___
Comments: _________________________


ONE STRENGTH:
_______________________________

ONE SUGGESTION:
_______________________________


Note: Be constructive and specific!
Your feedback helps your peers
improve their work.
'''

ax4.text(0.02, 0.98, form, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Evaluation Form', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
