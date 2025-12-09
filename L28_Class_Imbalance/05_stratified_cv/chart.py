"""Stratified Cross-Validation - Preserving class ratios"""
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
fig.suptitle('Stratified Cross-Validation for Imbalanced Data', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Problem with regular K-fold
ax1 = axes[0, 0]
ax1.axis('off')

# Visualize fold distributions
folds_regular = [
    [85, 15],  # Fold 1
    [95, 5],   # Fold 2
    [80, 20],  # Fold 3
    [90, 10],  # Fold 4
    [100, 0],  # Fold 5 - no minority!
]

for i, (maj, minn) in enumerate(folds_regular):
    y = 4 - i
    ax1.barh(y, maj, height=0.5, color=MLBLUE, edgecolor='black')
    ax1.barh(y, minn, left=maj, height=0.5, color=MLRED, edgecolor='black')
    ax1.text(105, y, f'{minn}% minority', fontsize=9, va='center')

ax1.set_yticks(range(5))
ax1.set_yticklabels([f'Fold {i+1}' for i in range(5)])
ax1.set_xlim(0, 130)
ax1.set_title('Regular K-Fold: Uneven Distribution', fontsize=11, fontweight='bold', color=MLRED)
ax1.set_xlabel('Percentage', fontsize=10)

ax1.text(50, -0.8, 'Fold 5 has NO minority samples!', fontsize=10, color=MLRED, ha='center', fontweight='bold')

# Plot 2: Stratified K-fold
ax2 = axes[0, 1]
ax2.axis('off')

folds_stratified = [
    [90, 10],  # Same ratio in all folds
    [90, 10],
    [90, 10],
    [90, 10],
    [90, 10],
]

for i, (maj, minn) in enumerate(folds_stratified):
    y = 4 - i
    ax2.barh(y, maj, height=0.5, color=MLBLUE, edgecolor='black')
    ax2.barh(y, minn, left=maj, height=0.5, color=MLRED, edgecolor='black')
    ax2.text(105, y, f'{minn}% minority', fontsize=9, va='center', color=MLGREEN)

ax2.set_yticks(range(5))
ax2.set_yticklabels([f'Fold {i+1}' for i in range(5)])
ax2.set_xlim(0, 130)
ax2.set_title('Stratified K-Fold: Consistent Distribution', fontsize=11, fontweight='bold', color=MLGREEN)
ax2.set_xlabel('Percentage', fontsize=10)

ax2.text(50, -0.8, 'All folds have same class ratio!', fontsize=10, color=MLGREEN, ha='center', fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MLBLUE, label='Majority'),
                   Patch(facecolor=MLRED, label='Minority')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Plot 3: Why stratification matters
ax3 = axes[1, 0]

folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
regular_f1 = [0.65, 0.45, 0.70, 0.55, 0.00]  # High variance, one is 0
stratified_f1 = [0.62, 0.58, 0.65, 0.60, 0.63]  # Consistent

x = np.arange(len(folds))
width = 0.35

bars1 = ax3.bar(x - width/2, regular_f1, width, label='Regular K-Fold', color=MLORANGE, edgecolor='black')
bars2 = ax3.bar(x + width/2, stratified_f1, width, label='Stratified K-Fold', color=MLGREEN, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(folds)
ax3.set_title('Minority F1 Score Across Folds', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Minority Class F1', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')

# Add mean lines
ax3.axhline(np.mean(regular_f1), color=MLORANGE, linestyle='--', alpha=0.7)
ax3.axhline(np.mean(stratified_f1), color=MLGREEN, linestyle='--', alpha=0.7)

ax3.text(4.5, np.mean(regular_f1)+0.03, f'Mean: {np.mean(regular_f1):.2f}', fontsize=8, color=MLORANGE)
ax3.text(4.5, np.mean(stratified_f1)+0.03, f'Mean: {np.mean(stratified_f1):.2f}', fontsize=8, color=MLGREEN)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
STRATIFIED CROSS-VALIDATION IN SKLEARN

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split
)

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # KEY: preserves class ratio
    random_state=42
)


# Stratified K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    score = f1_score(y_test, model.predict(X_test))
    scores.append(score)


# Easier: cross_val_score with stratified CV
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y,
    cv=StratifiedKFold(n_splits=5),  # Explicitly stratified
    scoring='f1'
)

# Default cv=5 in sklearn is already StratifiedKFold
# for classifiers! But explicit is better.
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
