"""Accuracy Problems - When accuracy misleads"""
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

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('The Accuracy Paradox: When Accuracy Misleads', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Imbalanced class distribution
ax1 = axes[0, 0]

classes = ['Normal\n(95%)', 'Fraud\n(5%)']
counts = [950, 50]

bars = ax1.bar(classes, counts, color=[MLBLUE, MLRED], edgecolor='black', linewidth=1)

ax1.set_title('Imbalanced Dataset: Fraud Detection', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Number of Samples', fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# Add percentages
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, count + 20, f'{count}\n({count/10:.0f}%)',
             ha='center', fontsize=12, fontweight='bold')

# Plot 2: Naive classifier beats accuracy
ax2 = axes[0, 1]

models = ['Always Predict\n"Normal"', 'Random\nGuess', 'Actual\nML Model']
accuracy = [0.95, 0.50, 0.92]
fraud_recall = [0.0, 0.50, 0.75]

x = np.arange(len(models))
width = 0.35

bars1 = ax2.bar(x - width/2, accuracy, width, label='Accuracy', color=MLBLUE, edgecolor='black')
bars2 = ax2.bar(x + width/2, fraud_recall, width, label='Fraud Recall', color=MLRED, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_title('The Accuracy Trap', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Score', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis='y')
ax2.set_ylim(0, 1.1)

# Highlight the paradox
ax2.annotate('95% accurate but\ncatches 0 fraud!', xy=(0, 0.95), xytext=(0.5, 0.75),
             fontsize=9, color=MLRED, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLRED))

ax2.annotate('92% accurate AND\ncatches 75% fraud!', xy=(2, 0.75), xytext=(1.5, 0.55),
             fontsize=9, color=MLGREEN, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

# Plot 3: Cost of errors
ax3 = axes[1, 0]
ax3.axis('off')

cost_analysis = '''
COST OF ERRORS IN FINANCE

SCENARIO: Fraud Detection
- 1000 transactions
- 50 actual frauds (5%)
- Average fraud = $10,000

MODEL A: "Always Normal" (95% accuracy)
----------------------------------------
TN = 950, FP = 0, FN = 50, TP = 0

Missed fraud cost = 50 x $10,000 = $500,000
False alarm cost = 0 x $100 = $0

Total Cost = $500,000


MODEL B: ML Model (92% accuracy)
---------------------------------
TN = 915, FP = 35, FN = 12, TP = 38

Missed fraud cost = 12 x $10,000 = $120,000
False alarm cost = 35 x $100 = $3,500

Total Cost = $123,500


CONCLUSION:
-----------
Model B has LOWER accuracy but saves $376,500!

Accuracy ignores:
1. Class imbalance
2. Different error costs
3. Business impact

ALWAYS consider the COST of errors,
not just the COUNT of errors.
'''

ax3.text(0.02, 0.98, cost_analysis, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Cost-Sensitive Analysis', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: When to use accuracy
ax4 = axes[1, 1]
ax4.axis('off')

guidelines = '''
WHEN IS ACCURACY APPROPRIATE?

USE ACCURACY WHEN:
------------------
[Y] Classes are balanced (50/50 or close)
[Y] All errors are equally costly
[Y] Quick, simple metric needed
[Y] Communicating to non-technical audience

DON'T USE ACCURACY ALONE WHEN:
------------------------------
[X] Imbalanced classes (fraud, disease, etc.)
[X] False positives and false negatives
    have different costs
[X] One class is much more important
[X] Need to tune decision threshold

BETTER ALTERNATIVES:
--------------------
- Precision: When false positives are costly
  (spam detection, medical tests)

- Recall: When false negatives are costly
  (fraud detection, cancer screening)

- F1-Score: Balance of precision and recall

- ROC-AUC: Overall discriminative ability

- Profit/Cost metric: Direct business impact

RULE OF THUMB:
--------------
"If your class ratio is more extreme than
 70/30, accuracy is probably misleading."
'''

ax4.text(0.02, 0.98, guidelines, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('When to Use Accuracy', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
