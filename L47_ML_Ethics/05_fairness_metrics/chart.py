"""Fairness Metrics - Measuring Model Fairness"""
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
fig.suptitle('Fairness Metrics: Measuring Model Fairness', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is fairness
ax1 = axes[0, 0]
ax1.axis('off')

fairness_def = '''
WHAT IS FAIRNESS IN ML?

DEFINITION:
-----------
A model is fair if it performs
equally well for different groups.


PROTECTED ATTRIBUTES:
---------------------
Characteristics that should NOT
affect predictions unfairly:
- Geographic region
- Company size
- Industry sector
- Data availability


FAIRNESS IN FINANCE:
--------------------
Should a model give equal quality
predictions for:
- Large cap vs small cap stocks?
- US vs international markets?
- Tech vs traditional sectors?


WHY IT MATTERS:
---------------
- Small cap stocks deserve good models
- Emerging markets need fair analysis
- All investors deserve quality insights


TYPES OF FAIRNESS:
------------------
1. Group Fairness:
   Equal performance across groups

2. Individual Fairness:
   Similar inputs -> similar outputs

3. Counterfactual Fairness:
   Would prediction change if only
   protected attribute changed?


KEY QUESTION:
-------------
"Does my model work equally well
 for all relevant subgroups?"
'''

ax1.text(0.02, 0.98, fairness_def, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is Fairness?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Fairness metrics comparison
ax2 = axes[0, 1]

groups = ['Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']
accuracy = [0.85, 0.82, 0.75, 0.68]
f1_score = [0.83, 0.80, 0.72, 0.65]

x = np.arange(len(groups))
width = 0.35

bars1 = ax2.bar(x - width/2, accuracy, width, label='Accuracy', color=MLBLUE, alpha=0.7)
bars2 = ax2.bar(x + width/2, f1_score, width, label='F1 Score', color=MLGREEN, alpha=0.7)

ax2.set_ylabel('Score')
ax2.set_xticks(x)
ax2.set_xticklabels(groups)
ax2.legend(fontsize=8)
ax2.set_ylim(0, 1)
ax2.axhline(y=0.80, color=MLRED, linestyle='--', label='Fairness threshold')
ax2.set_title('Performance by Market Cap Group', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3, axis='y')

# Highlight unfair groups
ax2.add_patch(plt.Rectangle((1.5, 0), 2, 0.80, facecolor=MLRED, alpha=0.1))
ax2.text(2.5, 0.05, 'Potential unfairness', fontsize=8, ha='center', color=MLRED)

# Plot 3: Fairness metrics formulas
ax3 = axes[1, 0]
ax3.axis('off')

metrics = '''
COMMON FAIRNESS METRICS

1. DEMOGRAPHIC PARITY:
----------------------
P(Positive | Group A) = P(Positive | Group B)

"Both groups should have same
 rate of positive predictions"

Example: Same % of BUY signals
for large and small cap stocks


2. EQUALIZED ODDS:
------------------
TPR(Group A) = TPR(Group B)
FPR(Group A) = FPR(Group B)

"Same true positive and false positive
 rates across groups"


3. EQUAL OPPORTUNITY:
---------------------
TPR(Group A) = TPR(Group B)

"If outcome is positive, equal chance
 of correct prediction for all groups"


4. PREDICTIVE PARITY:
---------------------
PPV(Group A) = PPV(Group B)

"Positive predictions equally likely
 to be correct for all groups"


CALCULATING IN PYTHON:
----------------------
from sklearn.metrics import confusion_matrix

# For each group
for group in groups:
    mask = (data['group'] == group)
    cm = confusion_matrix(y_true[mask], y_pred[mask])
    tpr = cm[1,1] / (cm[1,0] + cm[1,1])
    fpr = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"{group}: TPR={tpr:.2f}, FPR={fpr:.2f}")
'''

ax3.text(0.02, 0.98, metrics, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Fairness Metrics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Fairness testing checklist
ax4 = axes[1, 1]
ax4.axis('off')

# Draw checklist with visual indicators
ax4.text(0.5, 0.95, 'FAIRNESS TESTING CHECKLIST', fontsize=11, ha='center', fontweight='bold')

checklist = [
    ('Identify subgroups', 'What groups should be compared?', True),
    ('Calculate metrics per group', 'Accuracy, TPR, FPR for each', True),
    ('Compare across groups', 'Look for large differences', True),
    ('Set fairness threshold', 'Max acceptable difference (e.g., 10%)', True),
    ('Investigate disparities', 'Why does performance differ?', True),
    ('Consider trade-offs', 'Fairness vs overall accuracy', True),
    ('Document findings', 'Report in your project', True),
    ('Mitigate if needed', 'Rebalance, re-weight, or retrain', False)
]

for i, (item, desc, required) in enumerate(checklist):
    y = 0.85 - i * 0.1
    color = MLGREEN if required else MLORANGE
    ax4.add_patch(plt.Rectangle((0.02, y-0.035), 0.03, 0.05,
                                 facecolor=color, alpha=0.5, edgecolor='gray'))
    ax4.text(0.07, y, item, fontsize=9, fontweight='bold', va='center')
    ax4.text(0.5, y, desc, fontsize=8, va='center', color='gray')

ax4.text(0.5, 0.08, 'Green = Required for project | Orange = Advanced',
         fontsize=8, ha='center', style='italic')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Testing Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
