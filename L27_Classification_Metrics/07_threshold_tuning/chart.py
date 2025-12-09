"""Threshold Tuning - Optimizing the decision threshold"""
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
fig.suptitle('Threshold Tuning: Optimizing the Decision Point', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Metrics vs threshold
ax1 = axes[0, 0]

thresholds = np.linspace(0.1, 0.9, 50)

# Simulated metrics
precision = 0.5 + 0.45 * thresholds - 0.1 * thresholds**2 + np.random.normal(0, 0.02, 50)
recall = 0.95 - 0.55 * thresholds + np.random.normal(0, 0.02, 50)
f1 = 2 * precision * recall / (precision + recall + 0.001)
accuracy = 0.7 + 0.2 * np.sin((thresholds - 0.5) * np.pi) + np.random.normal(0, 0.02, 50)

ax1.plot(thresholds, precision, color=MLBLUE, linewidth=2, label='Precision')
ax1.plot(thresholds, recall, color=MLORANGE, linewidth=2, label='Recall')
ax1.plot(thresholds, f1, color=MLGREEN, linewidth=2, label='F1 Score')
ax1.plot(thresholds, accuracy, color=MLPURPLE, linewidth=2, label='Accuracy', linestyle='--')

# Mark optimal points
f1_opt_idx = np.argmax(f1)
ax1.scatter([thresholds[f1_opt_idx]], [f1[f1_opt_idx]], c=MLGREEN, s=150,
            marker='*', zorder=5, edgecolors='black', label=f'Best F1 (t={thresholds[f1_opt_idx]:.2f})')

ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.7, label='Default (0.5)')

ax1.set_title('Metrics vs Decision Threshold', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Threshold', fontsize=10)
ax1.set_ylabel('Score', fontsize=10)
ax1.legend(fontsize=8, loc='center right')
ax1.grid(alpha=0.3)

# Plot 2: Cost-based threshold selection
ax2 = axes[0, 1]

# Cost function: FP cost = $100, FN cost = $1000
fp_cost = 100
fn_cost = 1000

# Simulated FP and FN counts
fp_count = 100 * (1 - thresholds) + np.random.normal(0, 5, 50)
fn_count = 50 * thresholds + np.random.normal(0, 3, 50)

total_cost = fp_cost * fp_count + fn_cost * fn_count

ax2.plot(thresholds, fp_count * fp_cost / 1000, color=MLBLUE, linewidth=2, label=f'FP Cost ($100 each)')
ax2.plot(thresholds, fn_count * fn_cost / 1000, color=MLORANGE, linewidth=2, label=f'FN Cost ($1000 each)')
ax2.plot(thresholds, total_cost / 1000, color=MLRED, linewidth=3, label='Total Cost')

# Mark optimal
cost_opt_idx = np.argmin(total_cost)
ax2.scatter([thresholds[cost_opt_idx]], [total_cost[cost_opt_idx]/1000], c=MLGREEN, s=150,
            marker='*', zorder=5, edgecolors='black', label=f'Min Cost (t={thresholds[cost_opt_idx]:.2f})')

ax2.set_title('Cost-Sensitive Threshold Selection', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Threshold', fontsize=10)
ax2.set_ylabel('Cost ($K)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Finding optimal threshold code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
THRESHOLD TUNING STRATEGIES

# Strategy 1: Maximize F1
from sklearn.metrics import f1_score
y_prob = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best F1 threshold: {best_threshold:.2f}")


# Strategy 2: Maximize Youden's J (TPR - FPR)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f"Youden's J threshold: {best_threshold:.2f}")


# Strategy 3: Target specific recall
target_recall = 0.90
recalls = [(y_prob >= t).sum() / y_test.sum() for t in thresholds]
# Find threshold that achieves target recall
idx = np.argmin(np.abs(np.array(recalls) - target_recall))
threshold_90_recall = thresholds[idx]


# Strategy 4: Cost-sensitive
fp_cost, fn_cost = 100, 1000
costs = []
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    costs.append(fp * fp_cost + fn * fn_cost)
best_threshold = thresholds[np.argmin(costs)]
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Threshold Selection Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Summary of strategies
ax4 = axes[1, 1]
ax4.axis('off')

strategies = '''
THRESHOLD SELECTION SUMMARY

STRATEGY          | WHEN TO USE
------------------|--------------------------------
Default (0.5)     | Balanced classes, equal costs

Maximize F1       | Imbalanced classes,
                  | balance precision/recall

Youden's J        | Medical screening,
                  | maximize TPR - FPR

Target Recall     | Must catch X% of positives
                  | (e.g., 95% of fraud)

Target Precision  | Must have X% accuracy
                  | in positive predictions

Cost-based        | Different error costs
                  | (business optimization)


KEY INSIGHTS:
-------------
1. The default 0.5 is rarely optimal

2. Always tune on VALIDATION set
   (not test set!)

3. Consider business constraints:
   - Minimum recall requirements
   - Maximum false alarm rate
   - Total cost budget

4. Re-tune when class distribution
   changes in production
'''

ax4.text(0.02, 0.98, strategies, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Strategy Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
