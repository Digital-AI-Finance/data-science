"""Grid Search - Hyperparameter Tuning"""
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
fig.suptitle('Grid Search: Hyperparameter Tuning with Pipelines', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Grid search concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
GRID SEARCH CONCEPT

HYPERPARAMETERS:
----------------
Parameters set BEFORE training.
Not learned from data.

Examples:
- n_estimators in RandomForest
- C in LogisticRegression
- n_components in PCA


GRID SEARCH:
------------
Try ALL combinations of hyperparameter values.

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

Tries: (0.1, l1), (0.1, l2), (1, l1), (1, l2), ...
Total: 3 x 2 = 6 combinations


FOR PIPELINES:
--------------
Use step name prefix!

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', LogisticRegression())
])

param_grid = {
    'pca__n_components': [5, 10, 15],
    'clf__C': [0.1, 1, 10]
}
# Use double underscore: step__parameter
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Grid Search Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Grid visualization
ax2 = axes[0, 1]

# Create grid search results
C_values = [0.01, 0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1]

# Simulated accuracy scores
scores = np.array([
    [0.65, 0.72, 0.78, 0.76, 0.70],
    [0.70, 0.80, 0.85, 0.82, 0.75],
    [0.68, 0.78, 0.88, 0.85, 0.78],
    [0.60, 0.70, 0.80, 0.82, 0.80]
])

im = ax2.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=0.9)

ax2.set_xticks(range(len(C_values)))
ax2.set_xticklabels(C_values)
ax2.set_yticks(range(len(gamma_values)))
ax2.set_yticklabels(gamma_values)

ax2.set_xlabel('C (Regularization)')
ax2.set_ylabel('gamma')

# Annotate
for i in range(len(gamma_values)):
    for j in range(len(C_values)):
        color = 'white' if scores[i, j] > 0.8 else 'black'
        ax2.text(j, i, f'{scores[i, j]:.2f}', ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')

# Mark best
best_i, best_j = np.unravel_index(scores.argmax(), scores.shape)
ax2.scatter(best_j, best_i, s=300, facecolors='none', edgecolors=MLRED, linewidths=3)

ax2.set_title('Grid Search Results (CV Accuracy)', fontsize=11, fontweight='bold', color=MLPURPLE)
plt.colorbar(im, ax=ax2, shrink=0.8, label='Accuracy')

# Plot 3: GridSearchCV code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
GRIDSEARCHCV WITH PIPELINE

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svc', SVC())
])

# Define parameter grid (use step__param)
param_grid = {
    'pca__n_components': [5, 10, 15, 20],
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['rbf', 'linear'],
    'svc__gamma': ['scale', 'auto']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=5,                    # 5-fold CV
    scoring='accuracy',      # Metric to optimize
    n_jobs=-1,               # Use all CPU cores
    verbose=1,               # Print progress
    return_train_score=True  # Also track train scores
)

# Fit (tries all combinations)
grid_search.fit(X_train, y_train)

# Results
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Use best model
best_pipe = grid_search.best_estimator_
y_pred = best_pipe.predict(X_test)
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('GridSearchCV Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Results analysis
ax4 = axes[1, 1]
ax4.axis('off')

analysis = '''
ANALYZING GRID SEARCH RESULTS

# Full results DataFrame
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)


KEY COLUMNS:
------------
results['params']           # Parameter combinations
results['mean_test_score']  # Mean CV score
results['std_test_score']   # Std of CV scores
results['rank_test_score']  # Rank (1 = best)


BEST RESULTS:
-------------
grid_search.best_params_     # Best parameters
grid_search.best_score_      # Best mean CV score
grid_search.best_estimator_  # Fitted best model


OVERFITTING CHECK:
------------------
# Compare train vs test scores
train_scores = results['mean_train_score']
test_scores = results['mean_test_score']

# Large gap = overfitting


REFIT:
------
By default, GridSearchCV refits best model
on ALL training data.

grid_search.best_estimator_.predict(X_test)


CAUTION:
--------
Grid search = exhaustive = slow!
4 params x 5 values each = 625 combinations
x 5 folds = 3125 model fits!

Consider RandomizedSearchCV for large grids.
'''

ax4.text(0.02, 0.98, analysis, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Results Analysis', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
