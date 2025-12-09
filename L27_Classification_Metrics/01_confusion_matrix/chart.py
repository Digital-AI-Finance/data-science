"""Confusion Matrix - The foundation of classification metrics"""
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
fig.suptitle('The Confusion Matrix: Foundation of Classification Metrics', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Confusion matrix visualization
ax1 = axes[0, 0]

# Sample confusion matrix
cm = np.array([[85, 15], [10, 90]])
labels = ['Actual\nNegative', 'Actual\nPositive']
pred_labels = ['Predicted Negative', 'Predicted Positive']

im = ax1.imshow(cm, cmap='Blues', aspect='auto')

# Add text annotations
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > 50 else 'black'
        ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=16, fontweight='bold', color=color)

# Add cell labels
ax1.text(0, 0, '\nTN', ha='center', va='top', fontsize=10, color='white')
ax1.text(1, 0, '\nFP', ha='center', va='top', fontsize=10, color='white')
ax1.text(0, 1, '\nFN', ha='center', va='top', fontsize=10, color='white')
ax1.text(1, 1, '\nTP', ha='center', va='top', fontsize=10, color='white')

ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(pred_labels)
ax1.set_yticklabels(labels)
ax1.set_title('Confusion Matrix (2x2)', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.colorbar(im, ax=ax1, shrink=0.7)

# Plot 2: Terminology explanation
ax2 = axes[0, 1]
ax2.axis('off')

terminology = '''
CONFUSION MATRIX TERMINOLOGY

TRUE POSITIVE (TP) = 90
- Predicted Positive, Actually Positive
- Correctly identified positive cases
- "Hit"

TRUE NEGATIVE (TN) = 85
- Predicted Negative, Actually Negative
- Correctly identified negative cases
- "Correct rejection"

FALSE POSITIVE (FP) = 15
- Predicted Positive, Actually Negative
- Incorrectly flagged as positive
- "False alarm" / Type I Error

FALSE NEGATIVE (FN) = 10
- Predicted Negative, Actually Positive
- Missed positive cases
- "Miss" / Type II Error


LAYOUT:
          Predicted
          Neg    Pos
Actual  |-----|-----|
  Neg   | TN  | FP  |
  Pos   | FN  | TP  |
        |-----|-----|
'''

ax2.text(0.02, 0.98, terminology, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Terminology', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Metrics from confusion matrix
ax3 = axes[1, 0]
ax3.axis('off')

metrics = '''
METRICS DERIVED FROM CONFUSION MATRIX

             TN=85  FP=15
             FN=10  TP=90
             Total=200

ACCURACY
= (TP + TN) / Total
= (90 + 85) / 200 = 87.5%

PRECISION (Positive Predictive Value)
= TP / (TP + FP)
= 90 / (90 + 15) = 85.7%

RECALL (Sensitivity, True Positive Rate)
= TP / (TP + FN)
= 90 / (90 + 10) = 90.0%

SPECIFICITY (True Negative Rate)
= TN / (TN + FP)
= 85 / (85 + 15) = 85.0%

F1-SCORE
= 2 * (Precision * Recall) / (Precision + Recall)
= 2 * (0.857 * 0.900) / (0.857 + 0.900) = 87.8%
'''

ax3.text(0.02, 0.98, metrics, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Metrics Calculation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
CONFUSION MATRIX IN SKLEARN

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get predictions
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Output: [[85 15]
#          [10 90]]

# Access individual values
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Visual display
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Down', 'Up'],
    cmap='Blues',
    ax=ax
)
plt.title('Confusion Matrix')
plt.show()

# Or from estimator directly
ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test,
    display_labels=['Down', 'Up'],
    cmap='Blues'
)
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
