"""AUC Interpretation - Area Under the ROC Curve"""
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
fig.suptitle('AUC: Area Under the ROC Curve', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: AUC interpretation scale
ax1 = axes[0, 0]

# Create AUC scale
auc_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
interpretations = ['Random\nGuessing', 'Poor', 'Fair', 'Good', 'Excellent', 'Perfect']
colors = [MLRED, MLRED, MLORANGE, MLGREEN, MLBLUE, MLPURPLE]

# Create horizontal bar representation
for i, (auc, interp, color) in enumerate(zip(auc_values, interpretations, colors)):
    ax1.barh(0, 0.1, left=auc-0.05, color=color, edgecolor='black', height=0.5)
    ax1.text(auc, 0.4, f'{auc}', ha='center', fontsize=10, fontweight='bold')
    ax1.text(auc, -0.4, interp, ha='center', fontsize=8)

ax1.set_xlim(0.45, 1.05)
ax1.set_ylim(-0.8, 0.8)
ax1.set_yticks([])
ax1.set_xlabel('AUC Value', fontsize=10)
ax1.set_title('AUC Interpretation Scale', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add arrow
ax1.annotate('', xy=(1.0, 0.6), xytext=(0.5, 0.6),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax1.text(0.75, 0.65, 'Better', fontsize=10, ha='center')

# Plot 2: Probabilistic interpretation
ax2 = axes[0, 1]
ax2.axis('off')

interpretation = '''
AUC PROBABILISTIC INTERPRETATION

AUC = P(score(random positive) > score(random negative))

In plain English:
-----------------
"If you randomly pick one positive case and
 one negative case, AUC is the probability
 that the model ranks the positive case higher."


EXAMPLE (AUC = 0.85):
---------------------
Take 100 random pairs (1 positive, 1 negative).
In 85 of them, the positive will have a higher
predicted probability than the negative.


WHY THIS MATTERS:
-----------------
AUC measures RANKING ability, not calibration.

A model with AUC=0.9 means:
- It correctly orders positive vs negative
- Doesn't mean probabilities are accurate!

Two models can have:
- Same AUC
- Very different probability distributions


KEY PROPERTY:
-------------
AUC is THRESHOLD-INDEPENDENT.
It evaluates the model across ALL thresholds.
'''

ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Probabilistic Interpretation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Score distributions
ax3 = axes[1, 0]

# Generate score distributions for two classes
n = 500
scores_neg = np.random.beta(2, 5, n)  # Negative class
scores_pos = np.random.beta(5, 2, n)  # Positive class

ax3.hist(scores_neg, bins=30, alpha=0.6, color=MLBLUE, label='Actual Negative', density=True)
ax3.hist(scores_pos, bins=30, alpha=0.6, color=MLORANGE, label='Actual Positive', density=True)

# Mark threshold
threshold = 0.5
ax3.axvline(threshold, color=MLRED, linewidth=2.5, linestyle='--', label=f'Threshold={threshold}')

ax3.set_title('Score Distributions by Class (AUC~0.85)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Predicted Probability', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Add annotation
overlap = 0.3
ax3.annotate('More separation =\nHigher AUC', xy=(0.4, 1.5), fontsize=10,
             ha='center', color=MLPURPLE, fontweight='bold')

# Plot 4: AUC benchmarks
ax4 = axes[1, 1]

domains = ['Medical\nDiagnosis', 'Credit\nScoring', 'Fraud\nDetection', 'Marketing\nResponse', 'Stock\nDirection']
typical_auc = [0.85, 0.75, 0.95, 0.70, 0.55]
good_auc = [0.95, 0.85, 0.99, 0.80, 0.60]

x = np.arange(len(domains))
width = 0.35

bars1 = ax4.bar(x - width/2, typical_auc, width, label='Typical AUC', color=MLBLUE, edgecolor='black')
bars2 = ax4.bar(x + width/2, good_auc, width, label='Good AUC', color=MLGREEN, edgecolor='black')

ax4.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Random')

ax4.set_xticks(x)
ax4.set_xticklabels(domains, fontsize=9)
ax4.set_title('AUC Benchmarks by Domain', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_ylabel('AUC', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3, axis='y')
ax4.set_ylim(0.4, 1.05)

# Add note about stock prediction
ax4.annotate('Stock prediction\nis HARD!', xy=(4, 0.55), xytext=(3.5, 0.45),
             fontsize=9, color=MLRED, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLRED))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
