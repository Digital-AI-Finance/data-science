"""Evaluation Criteria - How Your Presentation Will Be Graded"""
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
fig.suptitle('Evaluation Criteria: How Your Project Will Be Graded', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Grade breakdown
ax1 = axes[0, 0]

categories = ['Problem\nDefinition\n(10%)', 'Data &\nEDA\n(15%)', 'Model\nDev\n(25%)',
              'Model\nEval\n(15%)', 'Deploy\n(15%)', 'Ethics\n(10%)', 'Present\n(10%)']
weights = [10, 15, 25, 15, 15, 10, 10]
colors = [MLBLUE, MLORANGE, MLGREEN, MLGREEN, MLPURPLE, MLRED, MLBLUE]

bars = ax1.bar(categories, weights, color=colors, alpha=0.7)
ax1.set_ylabel('Weight (%)')
ax1.set_ylim(0, 30)
ax1.set_title('Grade Distribution', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.grid(alpha=0.3, axis='y')

for bar, w in zip(bars, weights):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{w}%', ha='center', fontsize=10, fontweight='bold')

# Plot 2: Detailed rubric
ax2 = axes[0, 1]
ax2.axis('off')

rubric = '''
DETAILED EVALUATION RUBRIC

PROBLEM DEFINITION (10%):
-------------------------
5: Clear, well-motivated problem
4: Clear problem statement
3: Adequate problem definition
2: Vague or unclear
1: Missing or inappropriate


DATA & EDA (15%):
-----------------
5: Thorough EDA, great visualizations
4: Good EDA, clear insights
3: Basic EDA, some insights
2: Minimal exploration
1: No EDA or poor quality


MODEL DEVELOPMENT (25%):
------------------------
5: 3+ models, excellent engineering
4: 3+ models, good implementation
3: Minimum models, basic approach
2: Insufficient models
1: Model not working


MODEL EVALUATION (15%):
-----------------------
5: Comprehensive metrics, insights
4: Good evaluation, comparison
3: Basic metrics reported
2: Minimal evaluation
1: No evaluation


DEPLOYMENT (15%):
-----------------
5: Polished app, no errors
4: Working app, minor issues
3: Basic working app
2: Partially working
1: Not deployed


ETHICS (10%):
-------------
5: Thorough ethics discussion
4: Good limitations section
3: Basic acknowledgment
2: Minimal consideration
1: No ethics discussion


PRESENTATION (10%):
-------------------
5: Engaging, clear, great demo
4: Good delivery, working demo
3: Adequate presentation
2: Difficult to follow
1: Unprepared
'''

ax2.text(0.02, 0.98, rubric, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Detailed Rubric', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Grade scale visualization
ax3 = axes[1, 0]

grades = ['A', 'B', 'C', 'D', 'F']
ranges = ['90-100', '80-89', '70-79', '60-69', '<60']
descriptions = ['Exceptional', 'Good', 'Satisfactory', 'Needs Work', 'Incomplete']
colors = [MLGREEN, MLBLUE, MLORANGE, MLORANGE, MLRED]

# Create grade scale visual
for i, (grade, range_, desc, color) in enumerate(zip(grades, ranges, descriptions, colors)):
    ax3.add_patch(plt.Rectangle((0, 4-i-0.4), 0.15, 0.8, facecolor=color, alpha=0.7))
    ax3.text(0.075, 4-i, grade, fontsize=16, ha='center', va='center', fontweight='bold', color='white')
    ax3.text(0.2, 4-i+0.1, range_, fontsize=11, va='center', fontweight='bold')
    ax3.text(0.2, 4-i-0.15, desc, fontsize=9, va='center', color='gray')

ax3.set_xlim(0, 0.6)
ax3.set_ylim(-0.5, 4.5)
ax3.axis('off')
ax3.set_title('Grade Scale', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: What instructors look for
ax4 = axes[1, 1]
ax4.axis('off')

look_for = '''
WHAT INSTRUCTORS LOOK FOR

TECHNICAL COMPETENCE:
---------------------
- Correct use of methods
- Appropriate model choices
- Valid evaluation approach
- Clean, working code


CRITICAL THINKING:
------------------
- Justified decisions
- Awareness of limitations
- Honest about results
- Thoughtful analysis


COMMUNICATION:
--------------
- Clear explanations
- Logical structure
- Visual presentation
- Engaging delivery


EFFORT & COMPLETENESS:
----------------------
- All requirements met
- Thorough documentation
- Working deployment
- Practiced presentation


BONUS POINTS FOR:
-----------------
- Creative approach
- Extra features
- Strong visualizations
- Novel insights
- Exceptional presentation


COMMON MISTAKES:
----------------
- Overpromising results
- Ignoring limitations
- Poor time management
- Unprepared demo
- Going over time


REMEMBER:
---------
A good project honestly shows
what you learned, including
what didn't work!
'''

ax4.text(0.02, 0.98, look_for, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('What Instructors Look For', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
