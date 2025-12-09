"""Probability Output - Calibration and thresholds"""
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
fig.suptitle('Probability Outputs and Calibration', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Probability distribution
ax1 = axes[0, 0]

# Simulated probability outputs
n = 500
probs_class0 = np.random.beta(2, 5, n//2)  # Mostly low probabilities
probs_class1 = np.random.beta(5, 2, n//2)  # Mostly high probabilities

ax1.hist(probs_class0, bins=30, alpha=0.7, color=MLBLUE, label='True Class 0', edgecolor='black')
ax1.hist(probs_class1, bins=30, alpha=0.7, color=MLORANGE, label='True Class 1', edgecolor='black')
ax1.axvline(0.5, color=MLRED, linewidth=2.5, linestyle='--', label='Decision Threshold')

ax1.set_title('Distribution of P(Class 1) by True Class', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Predicted Probability P(Class 1)', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3, axis='y')

# Plot 2: Calibration curve
ax2 = axes[0, 1]

# Generate calibration data
n_bins = 10
bin_centers = np.linspace(0.05, 0.95, n_bins)
# Well-calibrated model (close to diagonal)
actual_probs = bin_centers + np.random.uniform(-0.05, 0.05, n_bins)
# Poorly calibrated model
poor_calib = np.clip(bin_centers * 1.3 - 0.15, 0, 1)

ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Perfect Calibration')
ax2.plot(bin_centers, actual_probs, color=MLBLUE, linewidth=2.5, marker='o', markersize=8,
         label='Logistic Regression')
ax2.plot(bin_centers, poor_calib, color=MLORANGE, linewidth=2.5, marker='s', markersize=8,
         label='Poor Calibration')

ax2.fill_between([0, 1], [0, 1], alpha=0.1, color=MLGREEN, label='Good calibration zone')

ax2.set_title('Calibration Plot (Reliability Diagram)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Mean Predicted Probability', fontsize=10)
ax2.set_ylabel('Fraction of Positives', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Plot 3: Threshold tuning
ax3 = axes[1, 0]

thresholds = np.linspace(0.1, 0.9, 50)

# Simulated metrics vs threshold
precision = 0.5 + 0.4 * thresholds + np.random.uniform(-0.02, 0.02, 50)
recall = 0.95 - 0.6 * thresholds + np.random.uniform(-0.02, 0.02, 50)
f1 = 2 * precision * recall / (precision + recall)

ax3.plot(thresholds, precision, color=MLBLUE, linewidth=2.5, label='Precision')
ax3.plot(thresholds, recall, color=MLORANGE, linewidth=2.5, label='Recall')
ax3.plot(thresholds, f1, color=MLGREEN, linewidth=2.5, label='F1-Score')

# Mark optimal F1
optimal_idx = np.argmax(f1)
ax3.scatter([thresholds[optimal_idx]], [f1[optimal_idx]], c=MLRED, s=150,
            marker='*', zorder=5, edgecolors='black', label=f'Optimal (t={thresholds[optimal_idx]:.2f})')
ax3.axvline(thresholds[optimal_idx], color='gray', linestyle='--', alpha=0.5)
ax3.axvline(0.5, color=MLPURPLE, linestyle=':', alpha=0.7, label='Default (0.5)')

ax3.set_title('Precision/Recall vs Decision Threshold', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Decision Threshold', fontsize=10)
ax3.set_ylabel('Score', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Code for probability usage
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Working with Probability Outputs

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# Get probabilities (not just predictions)
y_prob = model.predict_proba(X_test)[:, 1]

# Default prediction (threshold = 0.5)
y_pred_default = model.predict(X_test)

# Custom threshold (e.g., for high precision)
threshold = 0.7
y_pred_custom = (y_prob >= threshold).astype(int)

# Calibration check
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

# If poorly calibrated, use CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)
y_prob_calibrated = calibrated.predict_proba(X_test)[:, 1]

# Find optimal threshold for F1
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, (y_prob >= t).astype(int))
             for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {best_threshold:.2f}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
