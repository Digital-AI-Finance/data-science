"""Class Weights - Cost-sensitive learning"""
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
fig.suptitle('Class Weights: No Data Modification Needed', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Concept of class weights
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
CLASS WEIGHTS: THE IDEA

Instead of modifying data, penalize the model
more for errors on minority class.


UNWEIGHTED LOSS (imbalanced):
-----------------------------
Total Loss = sum of all errors

With 100 majority, 10 minority:
- Majority errors dominate loss
- Model ignores minority


WEIGHTED LOSS:
--------------
Total Loss = w_0 * (majority errors) +
             w_1 * (minority errors)

With weights {0: 1, 1: 10}:
- Minority errors count 10x more
- Model pays attention to minority!


HOW IT WORKS:
-------------
For logistic regression:
  Loss = -w_i * [y*log(p) + (1-y)*log(1-p)]

For trees:
  Gini impurity weighted by class weights


ADVANTAGE OVER RESAMPLING:
--------------------------
+ No synthetic data created
+ No data thrown away
+ Original distribution preserved
+ Faster training
+ Built into most sklearn classifiers
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Class Weights Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Effect of weights on decision boundary
ax2 = axes[0, 1]

# Generate imbalanced data
np.random.seed(42)
n_maj, n_min = 100, 10
X_maj = np.random.randn(n_maj, 2) * 1.5 + [-1, 0]
X_min = np.random.randn(n_min, 2) * 0.8 + [2, 2]

ax2.scatter(X_maj[:, 0], X_maj[:, 1], c=MLBLUE, s=30, alpha=0.5, label='Majority')
ax2.scatter(X_min[:, 0], X_min[:, 1], c=MLRED, s=80, alpha=1, label='Minority', edgecolors='black')

# Simulated decision boundaries
x_range = np.linspace(-4, 5, 100)

# Unweighted (biased toward majority)
y_unweighted = -x_range + 2.5
ax2.plot(x_range, y_unweighted, color=MLORANGE, linewidth=2.5, linestyle='--', label='Unweighted')

# Weighted (fair)
y_weighted = -x_range + 1.5
ax2.plot(x_range, y_weighted, color=MLGREEN, linewidth=2.5, label='class_weight="balanced"')

ax2.set_title('Decision Boundary with Class Weights', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1', fontsize=10)
ax2.set_ylabel('Feature 2', fontsize=10)
ax2.legend(fontsize=8)
ax2.set_xlim(-4, 5)
ax2.set_ylim(-4, 5)
ax2.grid(alpha=0.3)

# Plot 3: Performance comparison
ax3 = axes[1, 0]

methods = ['No weights', 'class_weight=\n"balanced"', 'Custom weights\n{0:1, 1:10}']
accuracy = [0.95, 0.85, 0.82]
minority_recall = [0.20, 0.75, 0.85]
f1_minority = [0.30, 0.70, 0.72]

x = np.arange(len(methods))
width = 0.25

bars1 = ax3.bar(x - width, accuracy, width, label='Accuracy', color=MLBLUE, edgecolor='black')
bars2 = ax3.bar(x, minority_recall, width, label='Minority Recall', color=MLRED, edgecolor='black')
bars3 = ax3.bar(x + width, f1_minority, width, label='Minority F1', color=MLGREEN, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(methods, fontsize=9)
ax3.set_title('Performance with Different Weight Settings', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Score', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')

# Highlight tradeoff
ax3.annotate('High accuracy,\npoor recall!', xy=(0, 0.20), xytext=(0.3, 0.35),
             fontsize=8, color=MLRED, arrowprops=dict(arrowstyle='->', color=MLRED))

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
CLASS WEIGHTS IN SKLEARN

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# Method 1: "balanced" (automatic)
# Weight = n_samples / (n_classes * n_samples_per_class)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)


# Method 2: Custom weights
# Higher weight = more penalty for errors
model = LogisticRegression(class_weight={0: 1, 1: 10})
model.fit(X_train, y_train)


# Method 3: Compute from data
weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
weight_dict = dict(zip(np.unique(y_train), weights))
print(f"Computed weights: {weight_dict}")


# Works with many classifiers:
# - LogisticRegression(class_weight=...)
# - RandomForestClassifier(class_weight=...)
# - SVC(class_weight=...)
# - DecisionTreeClassifier(class_weight=...)


# For sample-level weights:
sample_weights = np.where(y_train == 1, 10, 1)
model.fit(X_train, y_train, sample_weight=sample_weights)
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
