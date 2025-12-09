"""Cost-Sensitive Learning - Business-driven optimization"""
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
fig.suptitle('Cost-Sensitive Learning: Optimize for Business Impact', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Cost matrix
ax1 = axes[0, 0]

# Cost matrix visualization
cost_matrix = np.array([[0, 100], [10000, 0]])

im = ax1.imshow(cost_matrix, cmap='Reds', aspect='auto')

for i in range(2):
    for j in range(2):
        val = cost_matrix[i, j]
        color = 'white' if val > 5000 else 'black'
        ax1.text(j, i, f'${val:,}', ha='center', va='center', fontsize=14, fontweight='bold', color=color)

ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Predicted Normal', 'Predicted Fraud'])
ax1.set_yticklabels(['Actual Normal', 'Actual Fraud'])
ax1.set_title('Cost Matrix (Fraud Detection)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add cell labels
ax1.text(0, 0, '\n\nTN: $0', ha='center', fontsize=9)
ax1.text(1, 0, '\n\nFP: $100', ha='center', fontsize=9)
ax1.text(0, 1, '\n\nFN: $10,000', ha='center', fontsize=9, color='white')
ax1.text(1, 1, '\n\nTP: $0', ha='center', fontsize=9, color='white')

plt.colorbar(im, ax=ax1, shrink=0.7, label='Cost ($)')

# Plot 2: Cost-based threshold optimization
ax2 = axes[0, 1]

thresholds = np.linspace(0.01, 0.99, 100)

# Simulated FP, FN counts
fp_count = 100 * (1 - thresholds)**2
fn_count = 50 * thresholds**2

# Costs
fp_cost = 100
fn_cost = 10000

total_cost = fp_count * fp_cost + fn_count * fn_cost

ax2.plot(thresholds, fp_count * fp_cost / 1000, color=MLBLUE, linewidth=2, label='FP Cost')
ax2.plot(thresholds, fn_count * fn_cost / 1000, color=MLRED, linewidth=2, label='FN Cost')
ax2.plot(thresholds, total_cost / 1000, color=MLPURPLE, linewidth=3, label='Total Cost')

# Optimal threshold
opt_idx = np.argmin(total_cost)
ax2.scatter([thresholds[opt_idx]], [total_cost[opt_idx]/1000], c=MLGREEN, s=200, marker='*',
            zorder=5, edgecolors='black', label=f'Optimal t={thresholds[opt_idx]:.2f}')
ax2.axvline(thresholds[opt_idx], color='gray', linestyle='--', alpha=0.5)

ax2.axvline(0.5, color=MLORANGE, linestyle=':', alpha=0.7, label='Default t=0.5')

ax2.set_title('Cost vs Threshold', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Decision Threshold', fontsize=10)
ax2.set_ylabel('Cost ($K)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Expected profit calculation
ax3 = axes[1, 0]
ax3.axis('off')

calculation = '''
EXPECTED PROFIT CALCULATION

For each prediction, calculate expected profit:

E[Profit] = P(fraud) * Profit_if_fraud +
            P(normal) * Profit_if_normal


FRAUD DETECTION EXAMPLE:
------------------------
Costs:
- False Positive: $100 (investigation cost)
- False Negative: $10,000 (missed fraud loss)

For a transaction with P(fraud) = 0.3:

If we flag it:
  E[Profit] = 0.3 * $10,000 - 0.7 * $100 = $2,930

If we don't flag it:
  E[Profit] = 0.3 * (-$10,000) + 0.7 * $0 = -$3,000

Decision: FLAG IT (higher expected profit)


OPTIMAL THRESHOLD FORMULA:
--------------------------
Flag as fraud when:

P(fraud) > Cost_FP / (Cost_FP + Cost_FN)

With costs $100 and $10,000:
Threshold = 100 / (100 + 10000) = 0.0099

Flag transactions with P(fraud) > 1%!

(Much lower than default 50%)
'''

ax3.text(0.02, 0.98, calculation, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Expected Profit Calculation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
COST-SENSITIVE LEARNING IN PYTHON

import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_cost(y_true, y_pred, cost_fp=100, cost_fn=10000):
    """Calculate total cost of predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * cost_fp + fn * cost_fn


def find_optimal_threshold(y_true, y_prob, cost_fp=100, cost_fn=10000):
    """Find threshold that minimizes cost."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cost = calculate_cost(y_true, y_pred, cost_fp, cost_fn)
        costs.append(cost)

    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]


# Usage
y_prob = model.predict_proba(X_test)[:, 1]
best_t, min_cost = find_optimal_threshold(y_test, y_prob)
print(f"Optimal threshold: {best_t:.2f}")
print(f"Minimum cost: ${min_cost:,.0f}")

# Use optimal threshold for predictions
y_pred = (y_prob >= best_t).astype(int)


# Theoretical optimal threshold
t_optimal = cost_fp / (cost_fp + cost_fn)
print(f"Theoretical optimal: {t_optimal:.4f}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
