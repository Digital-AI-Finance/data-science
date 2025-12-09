"""ROC Curve - Receiver Operating Characteristic"""
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
fig.suptitle('ROC Curve: Receiver Operating Characteristic', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: ROC Curve
ax1 = axes[0, 0]

# Generate sample ROC curves
fpr_good = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])
tpr_good = np.array([0, 0.4, 0.6, 0.75, 0.85, 0.9, 0.95, 0.98, 1.0])

fpr_ok = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
tpr_ok = np.array([0, 0.25, 0.45, 0.6, 0.7, 0.78, 0.85, 0.93, 1.0])

# Plot ROC curves
ax1.plot(fpr_good, tpr_good, color=MLGREEN, linewidth=2.5, label='Good Model (AUC=0.92)')
ax1.plot(fpr_ok, tpr_ok, color=MLORANGE, linewidth=2.5, label='OK Model (AUC=0.78)')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random (AUC=0.50)')

ax1.fill_between(fpr_good, 0, tpr_good, alpha=0.2, color=MLGREEN)

ax1.set_title('ROC Curve', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('False Positive Rate (FPR)', fontsize=10)
ax1.set_ylabel('True Positive Rate (TPR)', fontsize=10)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Add annotations
ax1.annotate('Perfect\nClassifier', xy=(0, 1), xytext=(0.15, 0.85),
             fontsize=9, arrowprops=dict(arrowstyle='->', color=MLPURPLE))

# Plot 2: Understanding ROC
ax2 = axes[0, 1]
ax2.axis('off')

explanation = '''
UNDERSTANDING ROC CURVE

AXES:
-----
X-axis: False Positive Rate (FPR)
  = FP / (FP + TN) = 1 - Specificity
  "Of actual negatives, how many did we
   incorrectly flag as positive?"

Y-axis: True Positive Rate (TPR)
  = TP / (TP + FN) = Recall = Sensitivity
  "Of actual positives, how many did we
   correctly identify?"

KEY POINTS:
-----------
(0, 0): Predict all as Negative
(1, 1): Predict all as Positive
(0, 1): Perfect classifier
Diagonal: Random guessing

HOW IT'S CREATED:
-----------------
1. Sort samples by predicted probability
2. For each threshold (from 1 to 0):
   - Compute TPR and FPR
   - Plot the point
3. Connect all points

THRESHOLD EFFECT:
-----------------
High threshold (0.9): Few positives
  -> Low TPR, Low FPR (bottom-left)

Low threshold (0.1): Many positives
  -> High TPR, High FPR (top-right)
'''

ax2.text(0.02, 0.98, explanation, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Understanding ROC', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Multiple models comparison
ax3 = axes[1, 0]

# Different model ROC curves
models = {
    'Random Forest': (np.array([0, 0.03, 0.08, 0.15, 0.25, 0.4, 0.6, 1.0]),
                      np.array([0, 0.35, 0.55, 0.72, 0.85, 0.92, 0.97, 1.0]), MLGREEN),
    'Logistic Reg': (np.array([0, 0.05, 0.12, 0.2, 0.35, 0.5, 0.7, 1.0]),
                     np.array([0, 0.3, 0.48, 0.65, 0.78, 0.87, 0.94, 1.0]), MLBLUE),
    'Decision Tree': (np.array([0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]),
                      np.array([0, 0.25, 0.45, 0.58, 0.7, 0.8, 0.9, 1.0]), MLORANGE)
}

for name, (fpr, tpr, color) in models.items():
    # Approximate AUC
    auc = np.trapz(tpr, fpr)
    ax3.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name} (AUC={auc:.2f})')

ax3.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random')

ax3.set_title('ROC Comparison: Model Selection', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('False Positive Rate', fontsize=10)
ax3.set_ylabel('True Positive Rate', fontsize=10)
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
ROC CURVE IN SKLEARN

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# Get probability predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.4f}")

# Plot manually
plt.plot(fpr, tpr, label=f'Model (AUC={auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Or use sklearn's display
fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
plt.show()

# For cross-validation
from sklearn.model_selection import cross_val_predict
y_prob_cv = cross_val_predict(model, X, y, cv=5, method='predict_proba')
auc_cv = roc_auc_score(y, y_prob_cv[:, 1])
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
