"""Random Forest - Ensemble of trees"""
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
fig.suptitle('Random Forest: Ensemble of Decision Trees', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Random Forest concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
RANDOM FOREST: THE IDEA

"Wisdom of the Crowd"
---------------------
Many weak learners -> One strong learner

HOW IT WORKS:

1. BAGGING (Bootstrap Aggregating)
   - Create N random samples (with replacement)
   - Each sample is ~63% of original data
   - Each tree trains on different data

2. RANDOM FEATURE SELECTION
   - At each split, consider only sqrt(p) features
   - Different trees use different features
   - Decorrelates the trees

3. AGGREGATION
   - Classification: Majority vote
   - Regression: Average prediction


WHY IT WORKS:
-------------
- Individual trees overfit differently
- Averaging reduces variance (errors cancel)
- Bias remains similar
- Result: Lower overall error

"Random Forest is almost impossible to
 overfit by adding more trees"
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Random Forest Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Number of trees vs accuracy
ax2 = axes[0, 1]

n_trees = [1, 5, 10, 20, 50, 100, 200, 500]
test_accuracy = [0.72, 0.78, 0.82, 0.85, 0.87, 0.88, 0.885, 0.887]
train_accuracy = [1.0, 0.99, 0.98, 0.97, 0.96, 0.96, 0.96, 0.96]

ax2.plot(n_trees, train_accuracy, color=MLBLUE, linewidth=2.5, marker='o', markersize=6, label='Train Accuracy')
ax2.plot(n_trees, test_accuracy, color=MLORANGE, linewidth=2.5, marker='s', markersize=6, label='Test Accuracy')

ax2.axhline(test_accuracy[-1], color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(n_trees, 0.7, test_accuracy, alpha=0.1, color=MLORANGE)

ax2.set_xscale('log')
ax2.set_title('Effect of Number of Trees', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Number of Trees (n_estimators)', fontsize=10)
ax2.set_ylabel('Accuracy', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

ax2.annotate('Diminishing returns\nafter ~100 trees', xy=(200, 0.885), xytext=(50, 0.79),
             fontsize=9, arrowprops=dict(arrowstyle='->', color=MLPURPLE))

# Plot 3: sklearn code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
RANDOM FOREST IN SKLEARN

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(
    n_estimators=100,       # Number of trees
    max_depth=10,           # Max depth per tree
    min_samples_split=5,    # Min samples to split
    max_features='sqrt',    # Features per split
    bootstrap=True,         # Use bootstrap samples
    oob_score=True,         # Out-of-bag score
    n_jobs=-1,              # Use all CPU cores
    random_state=42
)

rf_clf.fit(X_train, y_train)

# Predictions
y_pred = rf_clf.predict(X_test)
y_prob = rf_clf.predict_proba(X_test)

# Out-of-bag score (free validation!)
print(f"OOB Score: {rf_clf.oob_score_:.4f}")

# Feature importance
importance = rf_clf.feature_importances_

# Regression
rf_reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Tree vs Forest comparison
ax4 = axes[1, 1]

methods = ['Single Tree', 'Random Forest\n(100 trees)']
train_scores = [0.98, 0.97]
test_scores = [0.75, 0.88]
variance = [0.08, 0.02]

x = np.arange(len(methods))
width = 0.25

bars1 = ax4.bar(x - width, train_scores, width, label='Train', color=MLBLUE, edgecolor='black')
bars2 = ax4.bar(x, test_scores, width, label='Test', color=MLORANGE, edgecolor='black')
bars3 = ax4.bar(x + width, variance, width, label='Variance', color=MLRED, edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(methods)
ax4.set_title('Decision Tree vs Random Forest', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_ylabel('Score', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3, axis='y')

# Add improvement annotation
ax4.annotate('+13% test\naccuracy!', xy=(1, test_scores[1]), xytext=(1.3, 0.92),
             fontsize=10, color=MLGREEN, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
