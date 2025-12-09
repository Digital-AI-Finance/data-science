"""Rubric Review - Understanding Grading Criteria"""
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
fig.suptitle('Project Rubric: Understanding How You Will Be Graded', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Grade breakdown
ax1 = axes[0, 0]

categories = [
    'Problem Definition\n(10%)',
    'Data & EDA\n(15%)',
    'Model Development\n(25%)',
    'Model Evaluation\n(15%)',
    'Deployment\n(15%)',
    'Ethics & Limits\n(10%)',
    'Presentation\n(10%)'
]
weights = [10, 15, 25, 15, 15, 10, 10]
colors = [MLBLUE, MLORANGE, MLGREEN, MLGREEN, MLPURPLE, MLRED, MLBLUE]

wedges, texts, autotexts = ax1.pie(weights, labels=categories, autopct='%1.0f%%',
                                    colors=colors, startangle=90,
                                    wedgeprops=dict(alpha=0.7))
ax1.set_title('Grade Distribution', fontsize=11, fontweight='bold', color=MLPURPLE)

for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')

# Plot 2: Detailed rubric
ax2 = axes[0, 1]
ax2.axis('off')

rubric_detail = '''
DETAILED GRADING CRITERIA

1. PROBLEM DEFINITION (10%)
---------------------------
- Clear problem statement
- Business/research motivation
- Defined success metrics
- Scope appropriateness

2. DATA & EDA (15%)
-------------------
- Data quality assessment
- Missing data handling
- Feature exploration
- Insightful visualizations
- Statistical summaries

3. MODEL DEVELOPMENT (25%)
--------------------------
- Minimum 3 different models
- Proper train/test split
- Feature engineering
- Hyperparameter tuning
- Code organization

4. MODEL EVALUATION (15%)
-------------------------
- Appropriate metrics used
- Cross-validation
- Model comparison
- Performance interpretation
- Honest assessment
'''

ax2.text(0.02, 0.98, rubric_detail, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Rubric Details (Part 1)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Rubric continued
ax3 = axes[1, 0]
ax3.axis('off')

rubric_detail2 = '''
DETAILED GRADING CRITERIA (CONTINUED)

5. DEPLOYMENT (15%)
-------------------
- Working Streamlit app
- Cloud deployment (Streamlit Cloud)
- No critical errors
- User-friendly interface
- Proper secrets handling

6. ETHICS & LIMITATIONS (10%)
-----------------------------
- Data bias discussion
- Model limitations acknowledged
- Fairness considerations
- Real-world implications
- Responsible AI practices

7. PRESENTATION (10%)
---------------------
- 5-7 minutes duration
- Clear structure
- Engaging delivery
- Live demo works
- Q&A handling


GRADE SCALE:
------------
A: 90-100%  - Exceptional work
B: 80-89%   - Good work
C: 70-79%   - Satisfactory
D: 60-69%   - Needs improvement
F: <60%     - Unsatisfactory


SUBMISSION REQUIREMENTS:
------------------------
- GitHub repository URL
- Deployed app URL
- Slides (PDF)
- All due before presentation
'''

ax3.text(0.02, 0.98, rubric_detail2, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Rubric Details (Part 2)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Grade levels visualization
ax4 = axes[1, 1]

criteria = ['Problem\nDefinition', 'Data &\nEDA', 'Model\nDev', 'Model\nEval', 'Deploy', 'Ethics', 'Present']
excellent = [9, 14, 23, 14, 14, 9, 9]
good = [7, 11, 18, 11, 11, 7, 7]
passing = [6, 9, 15, 9, 9, 6, 6]

x = np.arange(len(criteria))
width = 0.25

bars1 = ax4.bar(x - width, excellent, width, label='Excellent (90%+)', color=MLGREEN, alpha=0.7)
bars2 = ax4.bar(x, good, width, label='Good (80%+)', color=MLBLUE, alpha=0.7)
bars3 = ax4.bar(x + width, passing, width, label='Passing (70%+)', color=MLORANGE, alpha=0.7)

ax4.set_ylabel('Points')
ax4.set_xlabel('Criteria')
ax4.set_xticks(x)
ax4.set_xticklabels(criteria, fontsize=8)
ax4.legend(fontsize=8)
ax4.set_title('Points by Grade Level', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3, axis='y')

# Add max points line
max_points = [10, 15, 25, 15, 15, 10, 10]
ax4.plot(x, max_points, 'k--', linewidth=1, label='Max points')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
