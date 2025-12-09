"""Cross-Validation with Pipelines"""
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
fig.suptitle('Cross-Validation with ML Pipelines', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: CV with pipeline
ax1 = axes[0, 0]
ax1.axis('off')

cv_concept = '''
CROSS-VALIDATION WITH PIPELINES

WHY IT MATTERS:
---------------
Pipeline ensures preprocessing is done
INSIDE each CV fold, not before.

This prevents data leakage!


CORRECT APPROACH:
-----------------
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# Each fold: fit scaler on train, transform test
scores = cross_val_score(pipe, X, y, cv=5)


WRONG (leakage):
----------------
X_scaled = StandardScaler().fit_transform(X)
scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)
# Scaler saw all data including test folds!


THE DIFFERENCE:
---------------
Correct: Each fold has independent scaling
Wrong: All folds share the same scaling

Result: Wrong method overestimates performance!
'''

ax1.text(0.02, 0.98, cv_concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('CV with Pipeline', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: K-Fold visualization
ax2 = axes[0, 1]
ax2.axis('off')

# Draw K-fold diagram
n_folds = 5
fold_colors = [MLBLUE, MLBLUE, MLBLUE, MLBLUE, MLRED]

y_positions = np.linspace(0.85, 0.15, n_folds)
bar_height = 0.1

for fold_idx, y_pos in enumerate(y_positions):
    for i in range(n_folds):
        x_start = 0.1 + i * 0.16
        width = 0.14

        if i == fold_idx:
            color = MLRED
            label = 'Test' if fold_idx == 0 else ''
        else:
            color = MLBLUE
            label = 'Train' if (fold_idx == 0 and i == 1) else ''

        rect = plt.Rectangle((x_start, y_pos), width, bar_height,
                             facecolor=color, edgecolor='black', alpha=0.7)
        ax2.add_patch(rect)

    # Fold label
    ax2.text(0.05, y_pos + bar_height/2, f'Fold {fold_idx+1}',
             fontsize=9, va='center', fontweight='bold')

    # Pipeline label
    ax2.text(0.92, y_pos + bar_height/2, 'pipe.fit()\npipe.score()',
             fontsize=7, va='center', ha='left', fontfamily='monospace')

ax2.set_xlim(0, 1.1)
ax2.set_ylim(0, 1)
ax2.set_title('5-Fold Cross-Validation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Legend
ax2.text(0.3, 0.02, 'Train', fontsize=10, color=MLBLUE, fontweight='bold')
ax2.text(0.5, 0.02, 'Test', fontsize=10, color=MLRED, fontweight='bold')
ax2.text(0.7, 0.02, 'Pipeline refits each fold!', fontsize=9, style='italic')

# Plot 3: CV functions
ax3 = axes[1, 0]
ax3.axis('off')

cv_functions = '''
CROSS-VALIDATION FUNCTIONS

1. CROSS_VAL_SCORE (simple)
---------------------------
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    pipe, X, y,
    cv=5,              # Number of folds
    scoring='accuracy' # Metric
)
print(f"Mean: {scores.mean():.3f} +/- {scores.std():.3f}")


2. CROSS_VALIDATE (detailed)
----------------------------
from sklearn.model_selection import cross_validate

results = cross_validate(
    pipe, X, y,
    cv=5,
    scoring=['accuracy', 'f1', 'roc_auc'],
    return_train_score=True
)
print(results['test_accuracy'])
print(results['train_accuracy'])


3. CROSS_VAL_PREDICT (predictions)
----------------------------------
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(pipe, X, y, cv=5)
# Out-of-fold predictions for all samples
'''

ax3.text(0.02, 0.98, cv_functions, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('CV Functions', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: CV options
ax4 = axes[1, 1]
ax4.axis('off')

cv_options = '''
CV STRATEGY OPTIONS

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    LeaveOneOut,
    TimeSeriesSplit
)


KFOLD (basic):
--------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)


STRATIFIEDKFOLD (classification):
---------------------------------
cv = StratifiedKFold(n_splits=5)
# Preserves class distribution in each fold


REPEATEDKFOLD (more robust):
----------------------------
cv = RepeatedKFold(n_splits=5, n_repeats=10)
# 10 repetitions of 5-fold = 50 evaluations


TIMESERIESSPLIT (temporal data):
--------------------------------
cv = TimeSeriesSplit(n_splits=5)
# Respects time ordering, no look-ahead


USAGE:
------
scores = cross_val_score(pipe, X, y, cv=cv)


DEFAULT BEHAVIOR:
-----------------
cv=5 uses StratifiedKFold for classifiers
cv=5 uses KFold for regressors
'''

ax4.text(0.02, 0.98, cv_options, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('CV Strategy Options', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
