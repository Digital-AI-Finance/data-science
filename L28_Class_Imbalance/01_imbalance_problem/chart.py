"""Class Imbalance Problem - Why it matters"""
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
fig.suptitle('Class Imbalance: The Hidden Challenge', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Examples of imbalanced datasets
ax1 = axes[0, 0]

datasets = ['Fraud\nDetection', 'Medical\nDiagnosis', 'Spam\nDetection', 'Customer\nChurn', 'Balanced\n(Reference)']
minority_pct = [0.1, 2, 10, 15, 50]
majority_pct = [99.9, 98, 90, 85, 50]

x = np.arange(len(datasets))
width = 0.5

bars1 = ax1.bar(x, majority_pct, width, label='Majority Class', color=MLBLUE, edgecolor='black')
bars2 = ax1.bar(x, minority_pct, width, bottom=majority_pct, label='Minority Class', color=MLRED, edgecolor='black')

ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=9)
ax1.set_title('Class Distribution in Real-World Problems', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Percentage', fontsize=10)
ax1.legend(fontsize=9)
ax1.set_ylim(0, 105)

# Add ratio annotations
for i, (maj, min_) in enumerate(zip(majority_pct, minority_pct)):
    if min_ < 10:
        ratio = f'1:{int(maj/min_)}'
    else:
        ratio = f'1:{maj/min_:.1f}'
    ax1.text(i, 102, ratio, ha='center', fontsize=9, fontweight='bold')

# Plot 2: What goes wrong
ax2 = axes[0, 1]
ax2.axis('off')

problems = '''
WHY CLASS IMBALANCE IS A PROBLEM

THE ACCURACY TRAP:
------------------
Dataset: 99% Normal, 1% Fraud

"Always predict Normal" classifier:
- Accuracy: 99%
- Fraud detected: 0%

Looks great, catches nothing!


HOW MODELS LEARN:
-----------------
ML models minimize total errors.

With 99:1 imbalance:
- 99 ways to be "right" on majority
- 1 way to be "right" on minority

Models learn to ignore minority class!


WHAT FAILS:
-----------
[X] Decision boundaries favor majority
[X] Probability estimates are biased
[X] Default thresholds don't work
[X] Cross-validation gives false confidence
[X] Minority class patterns under-learned


SOLUTIONS (covered next):
-------------------------
- Resampling (over/under)
- SMOTE and variants
- Class weights
- Cost-sensitive learning
- Ensemble methods
'''

ax2.text(0.02, 0.98, problems, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('The Problem', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Decision boundary bias visualization
ax3 = axes[1, 0]

# Generate imbalanced data
np.random.seed(42)
n_majority = 200
n_minority = 10

X_maj = np.random.randn(n_majority, 2) * 1.5
X_min = np.random.randn(n_minority, 2) * 0.8 + [2, 2]

ax3.scatter(X_maj[:, 0], X_maj[:, 1], c=MLBLUE, s=30, alpha=0.5, label='Majority (200)', edgecolors='none')
ax3.scatter(X_min[:, 0], X_min[:, 1], c=MLRED, s=100, alpha=1, label='Minority (10)', edgecolors='black')

# Draw biased decision boundary (pushed toward minority)
x_line = np.linspace(-4, 5, 100)
y_line = -x_line + 3  # Biased toward minority
ax3.plot(x_line, y_line, color=MLORANGE, linewidth=2.5, linestyle='--', label='Biased boundary')

# Optimal boundary
y_optimal = -x_line + 2
ax3.plot(x_line, y_optimal, color=MLGREEN, linewidth=2.5, label='Optimal boundary')

ax3.set_title('Decision Boundary Bias', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Feature 1', fontsize=10)
ax3.set_ylabel('Feature 2', fontsize=10)
ax3.legend(fontsize=8, loc='upper right')
ax3.set_xlim(-4, 5)
ax3.set_ylim(-4, 5)
ax3.grid(alpha=0.3)

ax3.annotate('Boundary pushed\ntoward minority', xy=(1, 2), xytext=(-2, 3),
             fontsize=9, color=MLORANGE, arrowprops=dict(arrowstyle='->', color=MLORANGE))

# Plot 4: Imbalance ratios
ax4 = axes[1, 1]

ratios = ['1:1', '1:10', '1:100', '1:1000', '1:10000']
difficulty = [1, 2, 4, 7, 10]
typical_f1 = [0.90, 0.75, 0.55, 0.35, 0.15]

ax4.bar(ratios, difficulty, color=MLBLUE, alpha=0.5, label='Difficulty', edgecolor='black')
ax4.set_ylabel('Difficulty Level', color=MLBLUE, fontsize=10)
ax4.tick_params(axis='y', labelcolor=MLBLUE)

ax4_twin = ax4.twinx()
ax4_twin.plot(ratios, typical_f1, color=MLRED, linewidth=2.5, marker='o', markersize=8, label='Typical F1')
ax4_twin.set_ylabel('Typical Minority F1', color=MLRED, fontsize=10)
ax4_twin.tick_params(axis='y', labelcolor=MLRED)

ax4.set_title('Impact of Imbalance Ratio', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Imbalance Ratio (Minority:Majority)', fontsize=10)

# Add legend
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
