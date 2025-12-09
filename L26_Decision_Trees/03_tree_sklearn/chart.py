"""Decision Trees in sklearn"""
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
fig.suptitle('Decision Trees in sklearn', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic usage
ax1 = axes[0, 0]
ax1.axis('off')

basic_code = '''
BASIC DECISION TREE USAGE

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# For classification
clf = DecisionTreeClassifier(
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the model
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Probability predictions
y_prob = clf.predict_proba(X_test)

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)


For Regression:
---------------
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
'''

ax1.text(0.02, 0.98, basic_code, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Basic sklearn Usage', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Key parameters
ax2 = axes[0, 1]
ax2.axis('off')

params = '''
KEY PARAMETERS

DecisionTreeClassifier(
    criterion='gini',      # 'gini' or 'entropy'
    max_depth=None,        # Maximum tree depth
    min_samples_split=2,   # Min samples to split
    min_samples_leaf=1,    # Min samples in leaf
    max_features=None,     # Features per split
    random_state=42,       # Reproducibility
    class_weight=None      # Handle imbalance
)

CONTROLLING COMPLEXITY:
-----------------------
max_depth: Limit tree depth (most important!)
- None: Grow until pure
- 3-10: Good starting range

min_samples_split: Min samples to consider split
- Higher = simpler tree
- Try: 2, 5, 10, 20

min_samples_leaf: Min samples in final leaf
- Higher = simpler tree
- Try: 1, 5, 10

max_features: Features to consider per split
- None: all features
- 'sqrt': sqrt(n_features)
- 'log2': log2(n_features)
- int: exact number
'''

ax2.text(0.02, 0.98, params, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Key Parameters', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Effect of max_depth
ax3 = axes[1, 0]

depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, None]
depth_labels = [str(d) if d else 'None' for d in depths]

# Simulated accuracies
train_acc = [0.58, 0.65, 0.72, 0.78, 0.83, 0.87, 0.91, 0.94, 0.97, 0.99, 1.0, 1.0]
test_acc = [0.55, 0.63, 0.70, 0.74, 0.76, 0.77, 0.76, 0.74, 0.72, 0.68, 0.65, 0.62]

x = np.arange(len(depths))
ax3.plot(x, train_acc, color=MLBLUE, linewidth=2.5, marker='o', markersize=6, label='Train Accuracy')
ax3.plot(x, test_acc, color=MLORANGE, linewidth=2.5, marker='s', markersize=6, label='Test Accuracy')

# Mark optimal
optimal_idx = np.argmax(test_acc)
ax3.scatter([optimal_idx], [test_acc[optimal_idx]], c=MLGREEN, s=200,
            marker='*', zorder=5, edgecolors='black', label='Optimal Depth')
ax3.axvline(optimal_idx, color='gray', linestyle='--', alpha=0.5)

ax3.set_xticks(x)
ax3.set_xticklabels(depth_labels, fontsize=8)
ax3.set_title('Effect of max_depth on Performance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('max_depth', fontsize=10)
ax3.set_ylabel('Accuracy', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Add zones
ax3.text(1.5, 0.58, 'Underfitting', fontsize=9, color=MLRED)
ax3.text(9, 0.63, 'Overfitting', fontsize=9, color=MLRED)

# Plot 4: Model attributes
ax4 = axes[1, 1]
ax4.axis('off')

attributes = '''
USEFUL MODEL ATTRIBUTES

After fitting:
--------------
clf.tree_                  # Tree structure object
clf.feature_importances_   # Feature importance scores
clf.n_features_in_         # Number of features
clf.n_classes_             # Number of classes
clf.classes_               # Class labels
clf.max_depth              # Maximum depth reached

Tree inspection:
----------------
clf.tree_.node_count       # Number of nodes
clf.tree_.max_depth        # Actual max depth used
clf.tree_.feature          # Feature used at each node
clf.tree_.threshold        # Threshold at each node
clf.tree_.n_node_samples   # Samples at each node

Visualization:
--------------
from sklearn.tree import plot_tree, export_text

# Visual plot
plot_tree(clf, feature_names=['f1','f2'],
          class_names=['Down','Up'], filled=True)

# Text representation
print(export_text(clf, feature_names=['f1','f2']))
'''

ax4.text(0.02, 0.98, attributes, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Model Attributes', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
