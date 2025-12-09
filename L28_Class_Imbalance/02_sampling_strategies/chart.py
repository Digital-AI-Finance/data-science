"""Sampling Strategies - Over and under sampling"""
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
fig.suptitle('Sampling Strategies for Imbalanced Data', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Original imbalanced data
ax1 = axes[0, 0]

n_maj, n_min = 100, 10
X_maj = np.random.randn(n_maj, 2) * 1.5 + [-1, 0]
X_min = np.random.randn(n_min, 2) * 0.8 + [2, 2]

ax1.scatter(X_maj[:, 0], X_maj[:, 1], c=MLBLUE, s=40, alpha=0.6, label=f'Majority ({n_maj})', edgecolors='black')
ax1.scatter(X_min[:, 0], X_min[:, 1], c=MLRED, s=80, alpha=1, label=f'Minority ({n_min})', edgecolors='black')

ax1.set_title('Original: Imbalanced (100:10)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Feature 1', fontsize=10)
ax1.set_ylabel('Feature 2', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Random Oversampling
ax2 = axes[0, 1]

# Oversample minority
oversample_idx = np.random.choice(n_min, size=n_maj-n_min, replace=True)
X_min_over = np.vstack([X_min, X_min[oversample_idx] + np.random.randn(n_maj-n_min, 2)*0.1])

ax2.scatter(X_maj[:, 0], X_maj[:, 1], c=MLBLUE, s=40, alpha=0.6, label=f'Majority ({n_maj})', edgecolors='black')
ax2.scatter(X_min_over[:, 0], X_min_over[:, 1], c=MLRED, s=40, alpha=0.6, label=f'Minority ({len(X_min_over)})', edgecolors='black')

# Highlight duplicates
ax2.scatter(X_min[:, 0], X_min[:, 1], c='yellow', s=100, alpha=1, marker='*', edgecolors='black', label='Original')

ax2.set_title('Random Oversampling (100:100)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1', fontsize=10)
ax2.set_ylabel('Feature 2', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax2.text(0.02, 0.02, 'Duplicates exact points\n(Risk of overfitting)', transform=ax2.transAxes,
         fontsize=9, color=MLRED, style='italic')

# Plot 3: Random Undersampling
ax3 = axes[1, 0]

# Undersample majority
undersample_idx = np.random.choice(n_maj, size=n_min, replace=False)
X_maj_under = X_maj[undersample_idx]

ax3.scatter(X_maj_under[:, 0], X_maj_under[:, 1], c=MLBLUE, s=80, alpha=0.8, label=f'Majority ({len(X_maj_under)})', edgecolors='black')
ax3.scatter(X_min[:, 0], X_min[:, 1], c=MLRED, s=80, alpha=1, label=f'Minority ({n_min})', edgecolors='black')

# Show removed points (faded)
removed_idx = np.setdiff1d(range(n_maj), undersample_idx)
ax3.scatter(X_maj[removed_idx, 0], X_maj[removed_idx, 1], c='gray', s=20, alpha=0.2, label='Removed')

ax3.set_title('Random Undersampling (10:10)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Feature 1', fontsize=10)
ax3.set_ylabel('Feature 2', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

ax3.text(0.02, 0.02, 'Loses majority information\n(May lose important patterns)', transform=ax3.transAxes,
         fontsize=9, color=MLORANGE, style='italic')

# Plot 4: Comparison summary
ax4 = axes[1, 1]
ax4.axis('off')

comparison = '''
SAMPLING STRATEGIES COMPARISON

RANDOM OVERSAMPLING
-------------------
How: Duplicate minority samples
Pros:
  + No information loss
  + Simple to implement
Cons:
  - Risk of overfitting (exact copies)
  - Increases training time
  - Can memorize noise


RANDOM UNDERSAMPLING
--------------------
How: Remove majority samples
Pros:
  + Faster training
  + Simple to implement
Cons:
  - Loses information
  - May remove important patterns
  - Unstable results


COMBINED APPROACH
-----------------
Over-sample minority + Under-sample majority
Often works better than either alone!


SKLEARN (imbalanced-learn):
---------------------------
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
'''

ax4.text(0.02, 0.98, comparison, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Strategy Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
