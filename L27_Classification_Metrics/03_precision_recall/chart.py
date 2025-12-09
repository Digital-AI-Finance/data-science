"""Precision and Recall - The trade-off"""
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
fig.suptitle('Precision and Recall: The Fundamental Trade-off', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Visual explanation with Venn-style diagram
ax1 = axes[0, 0]
ax1.axis('off')

# Draw circles to represent predictions and actual positives
from matplotlib.patches import Circle, Rectangle

# Predicted positives (blue)
pred_circle = Circle((0.4, 0.5), 0.25, color=MLBLUE, alpha=0.4, label='Predicted Positive')
# Actual positives (orange)
actual_circle = Circle((0.55, 0.5), 0.25, color=MLORANGE, alpha=0.4, label='Actual Positive')

ax1.add_patch(pred_circle)
ax1.add_patch(actual_circle)

# Labels
ax1.text(0.3, 0.5, 'FP', fontsize=14, fontweight='bold', ha='center', va='center')
ax1.text(0.475, 0.5, 'TP', fontsize=14, fontweight='bold', ha='center', va='center')
ax1.text(0.65, 0.5, 'FN', fontsize=14, fontweight='bold', ha='center', va='center')

# Annotations
ax1.annotate('Precision = TP / (TP + FP)\n"Of predicted positives,\nhow many are correct?"',
             xy=(0.4, 0.15), fontsize=9, ha='center', color=MLBLUE, fontweight='bold')
ax1.annotate('Recall = TP / (TP + FN)\n"Of actual positives,\nhow many did we find?"',
             xy=(0.55, 0.85), fontsize=9, ha='center', color=MLORANGE, fontweight='bold')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('Visual: Precision vs Recall', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add legend
ax1.text(0.05, 0.95, 'Blue: Predicted Positive\nOrange: Actual Positive', fontsize=8,
         transform=ax1.transAxes, va='top')

# Plot 2: Precision-Recall trade-off curve
ax2 = axes[0, 1]

# Simulated precision-recall curve
recall = np.linspace(0, 1, 100)
precision = 1 - 0.5 * recall + 0.3 * np.sin(recall * np.pi) * (1 - recall)
precision = np.clip(precision, 0, 1)

ax2.plot(recall, precision, color=MLPURPLE, linewidth=2.5)
ax2.fill_between(recall, 0, precision, alpha=0.2, color=MLPURPLE)

# Mark points
points = [(0.2, 0.92), (0.5, 0.80), (0.8, 0.55), (0.95, 0.35)]
labels = ['High Prec\nLow Rec', 'Balanced', 'High Rec\nLow Prec', 'Catch All']
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]

for (r, p), label, color in zip(points, labels, colors):
    ax2.scatter([r], [p], s=150, c=color, edgecolors='black', zorder=5)
    ax2.annotate(label, xy=(r, p), xytext=(r+0.05, p+0.05), fontsize=8)

ax2.set_title('Precision-Recall Trade-off', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Recall', fontsize=10)
ax2.set_ylabel('Precision', fontsize=10)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.grid(alpha=0.3)

# Add baseline
ax2.axhline(0.3, color='gray', linestyle='--', linewidth=1.5, label='Random baseline')
ax2.legend(fontsize=8)

# Plot 3: When to prioritize each
ax3 = axes[1, 0]
ax3.axis('off')

priorities = '''
WHEN TO PRIORITIZE PRECISION vs RECALL

PRIORITIZE PRECISION (minimize FP):
-----------------------------------
- Email spam detection
  "Don't send important emails to spam!"

- Recommendation systems
  "Don't recommend irrelevant products!"

- Legal document classification
  "Don't mis-classify legal documents!"

PRIORITIZE RECALL (minimize FN):
--------------------------------
- Fraud detection
  "Don't miss any fraud, even if false alarms!"

- Cancer screening
  "Don't miss any cancer cases!"

- System intrusion detection
  "Don't miss any security threats!"

FINANCE EXAMPLES:
-----------------
High Precision:
- Trade signal generation (quality over quantity)
- High-confidence alerts only

High Recall:
- Risk monitoring (catch all potential issues)
- Compliance screening (don't miss violations)

THE KEY QUESTION:
-----------------
"What's worse: false alarm or missed case?"
'''

ax3.text(0.02, 0.98, priorities, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('When to Prioritize Each', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
PRECISION AND RECALL IN SKLEARN

from sklearn.metrics import (
    precision_score, recall_score,
    precision_recall_curve,
    PrecisionRecallDisplay
)

# Get predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(
    y_test, y_prob
)

# Plot PR curve
fig, ax = plt.subplots()
PrecisionRecallDisplay.from_predictions(
    y_test, y_prob, ax=ax, name='Model'
)
plt.title('Precision-Recall Curve')
plt.show()

# For multiclass
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
