"""Precision-Recall Tradeoff for Imbalanced Data"""
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
fig.suptitle('Precision-Recall Tradeoff in Imbalanced Settings', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: PR curve comparison (imbalanced vs balanced)
ax1 = axes[0, 0]

# Simulated PR curves
recall = np.linspace(0, 1, 100)

# Balanced data PR curve
pr_balanced = 0.9 - 0.4 * recall + 0.3 * np.sin(recall * np.pi) * (1 - recall)
pr_balanced = np.clip(pr_balanced, 0.1, 1)

# Imbalanced data PR curve (much lower)
pr_imbalanced = 0.5 - 0.35 * recall + 0.2 * np.sin(recall * np.pi) * (1 - recall)
pr_imbalanced = np.clip(pr_imbalanced, 0.02, 1)

ax1.plot(recall, pr_balanced, color=MLGREEN, linewidth=2.5, label='Balanced data (50/50)')
ax1.plot(recall, pr_imbalanced, color=MLRED, linewidth=2.5, label='Imbalanced data (99/1)')

# Baselines
ax1.axhline(0.5, color=MLGREEN, linestyle='--', alpha=0.5, label='Random (balanced)')
ax1.axhline(0.01, color=MLRED, linestyle='--', alpha=0.5, label='Random (imbalanced)')

ax1.set_title('PR Curves: Balanced vs Imbalanced', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Recall', fontsize=10)
ax1.set_ylabel('Precision', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

ax1.text(0.5, 0.08, 'Same model, different\nclass distributions!', fontsize=9, ha='center', color=MLPURPLE)

# Plot 2: Why ROC can be misleading
ax2 = axes[0, 1]
ax2.axis('off')

explanation = '''
WHY PR CURVES MATTER FOR IMBALANCED DATA

ROC CURVE PROBLEM:
------------------
ROC uses FPR = FP / (FP + TN)

With 99% negatives:
- TN is huge (990 samples)
- FP of 10 seems small: FPR = 10/990 = 1%
- But 10 false alarms might be too many!

ROC looks great even with poor precision.


PR CURVE ADVANTAGE:
-------------------
Precision = TP / (TP + FP)

Directly measures:
"Of my positive predictions, how many are right?"

This is what we care about in:
- Fraud detection
- Medical diagnosis
- Anomaly detection


BASELINE COMPARISON:
--------------------
ROC: Random baseline is always 50% (diagonal)

PR: Random baseline = positive class rate
    For 1% positive: baseline is 1%!

A model with 10% precision on 1% positive data
is 10x better than random!


USE PR-AUC (Average Precision):
-------------------------------
More informative than ROC-AUC for imbalanced data.
'''

ax2.text(0.02, 0.98, explanation, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('PR vs ROC for Imbalanced Data', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Threshold selection visualization
ax3 = axes[1, 0]

thresholds = np.linspace(0.1, 0.9, 50)

# Simulated metrics for imbalanced data
precision = 0.1 + 0.8 * thresholds
recall = 0.95 - 0.85 * thresholds
f1 = 2 * precision * recall / (precision + recall + 0.001)

ax3.plot(thresholds, precision, color=MLBLUE, linewidth=2, label='Precision')
ax3.plot(thresholds, recall, color=MLORANGE, linewidth=2, label='Recall')
ax3.plot(thresholds, f1, color=MLGREEN, linewidth=2.5, label='F1 Score')

# Mark key points
ax3.axvline(0.5, color='gray', linestyle=':', alpha=0.7, label='Default (0.5)')

# Optimal F1
opt_idx = np.argmax(f1)
ax3.scatter([thresholds[opt_idx]], [f1[opt_idx]], c=MLGREEN, s=150, marker='*', zorder=5, edgecolors='black')

# High recall point
high_recall_idx = np.argmin(np.abs(recall - 0.90))
ax3.scatter([thresholds[high_recall_idx]], [f1[high_recall_idx]], c=MLORANGE, s=100, marker='o', zorder=5, edgecolors='black')

ax3.set_title('Threshold Selection for Imbalanced Data', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Decision Threshold', fontsize=10)
ax3.set_ylabel('Score', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

ax3.annotate(f'Best F1\nt={thresholds[opt_idx]:.2f}', xy=(thresholds[opt_idx], f1[opt_idx]),
             xytext=(thresholds[opt_idx]+0.1, f1[opt_idx]+0.1), fontsize=8,
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

# Plot 4: Average Precision code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
PRECISION-RECALL IN SKLEARN

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay
)

# Get probability predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate PR curve
precisions, recalls, thresholds = precision_recall_curve(
    y_test, y_prob
)

# Average Precision (area under PR curve)
ap = average_precision_score(y_test, y_prob)
print(f"Average Precision: {ap:.4f}")

# Compare to positive class rate
baseline = y_test.mean()
print(f"Random baseline: {baseline:.4f}")
print(f"Lift over random: {ap/baseline:.1f}x")


# Plot PR curve
fig, ax = plt.subplots()
PrecisionRecallDisplay.from_predictions(
    y_test, y_prob,
    name='Model',
    ax=ax
)
ax.axhline(baseline, linestyle='--', label='Random')
plt.legend()
plt.title(f'PR Curve (AP={ap:.3f})')
plt.show()


# Use AP for model selection (better than ROC-AUC)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='average_precision')
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
