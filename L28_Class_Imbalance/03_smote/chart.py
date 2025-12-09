"""SMOTE - Synthetic Minority Over-sampling Technique"""
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
fig.suptitle('SMOTE: Synthetic Minority Over-sampling Technique', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: SMOTE algorithm visualization
ax1 = axes[0, 0]

# Original minority points
X_min = np.array([[2, 2], [3, 3], [2.5, 3.5], [3.5, 2.5], [4, 3]])

ax1.scatter(X_min[:, 0], X_min[:, 1], c=MLRED, s=150, marker='o', label='Original minority', edgecolors='black', zorder=5)

# Show k-nearest neighbors for one point
point_idx = 0
distances = np.sqrt(np.sum((X_min - X_min[point_idx])**2, axis=1))
k_nearest = np.argsort(distances)[1:4]  # k=3

# Draw lines to neighbors
for nn_idx in k_nearest:
    ax1.plot([X_min[point_idx, 0], X_min[nn_idx, 0]],
             [X_min[point_idx, 1], X_min[nn_idx, 1]],
             'g--', linewidth=2, alpha=0.7)

# Generate synthetic points along lines
synthetic_points = []
for nn_idx in k_nearest:
    alpha = np.random.uniform(0.3, 0.7)
    new_point = X_min[point_idx] + alpha * (X_min[nn_idx] - X_min[point_idx])
    synthetic_points.append(new_point)

synthetic_points = np.array(synthetic_points)
ax1.scatter(synthetic_points[:, 0], synthetic_points[:, 1], c=MLGREEN, s=100, marker='*',
            label='Synthetic (SMOTE)', edgecolors='black', zorder=4)

ax1.scatter([X_min[point_idx, 0]], [X_min[point_idx, 1]], c='yellow', s=200, marker='o',
            edgecolors='black', linewidths=2, zorder=6, label='Selected point')

ax1.set_title('SMOTE: Creating Synthetic Points', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Feature 1', fontsize=10)
ax1.set_ylabel('Feature 2', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: SMOTE algorithm steps
ax2 = axes[0, 1]
ax2.axis('off')

algorithm = '''
SMOTE ALGORITHM

For each minority sample x:
1. Find k nearest neighbors (typically k=5)
2. Randomly select one neighbor nn
3. Create synthetic point:

   x_new = x + rand(0,1) * (nn - x)

4. Repeat until desired balance


EXAMPLE:
--------
Point x = [2, 2]
Neighbor nn = [3, 3]
Random alpha = 0.4

x_new = [2, 2] + 0.4 * ([3, 3] - [2, 2])
x_new = [2, 2] + 0.4 * [1, 1]
x_new = [2.4, 2.4]


KEY INSIGHT:
------------
Creates NEW points along the line between
existing minority points.

- Not exact copies (avoids overfitting)
- Stays within minority "region"
- Preserves minority characteristics


VARIANTS:
---------
SMOTE       : Original algorithm
SMOTE-ENN   : SMOTE + Edited Nearest Neighbors
SMOTE-Tomek : SMOTE + Tomek links removal
ADASYN      : Adaptive Synthetic Sampling
Borderline-SMOTE : Focus on decision boundary
'''

ax2.text(0.02, 0.98, algorithm, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Algorithm Steps', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Before and After SMOTE
ax3 = axes[1, 0]

# Imbalanced data
n_maj, n_min = 80, 10
X_maj = np.random.randn(n_maj, 2) * 1.5 + [-1, 0]
X_min_orig = np.random.randn(n_min, 2) * 0.8 + [2, 2]

ax3.scatter(X_maj[:, 0], X_maj[:, 1], c=MLBLUE, s=30, alpha=0.5, label=f'Majority ({n_maj})')

# Generate SMOTE-like points
n_synthetic = n_maj - n_min
synthetic = []
for _ in range(n_synthetic):
    idx1, idx2 = np.random.choice(n_min, 2, replace=False)
    alpha = np.random.uniform(0, 1)
    new_point = X_min_orig[idx1] + alpha * (X_min_orig[idx2] - X_min_orig[idx1])
    synthetic.append(new_point)

synthetic = np.array(synthetic)
all_minority = np.vstack([X_min_orig, synthetic])

ax3.scatter(X_min_orig[:, 0], X_min_orig[:, 1], c=MLRED, s=80, alpha=1, label=f'Original minority ({n_min})', edgecolors='black')
ax3.scatter(synthetic[:, 0], synthetic[:, 1], c=MLGREEN, s=40, alpha=0.7, label=f'Synthetic ({n_synthetic})', marker='*')

ax3.set_title('After SMOTE: Balanced Dataset', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Feature 1', fontsize=10)
ax3.set_ylabel('Feature 2', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
SMOTE IN PYTHON (imbalanced-learn)

# Install: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

# Basic SMOTE
smote = SMOTE(
    sampling_strategy='auto',  # Balance to 1:1
    k_neighbors=5,             # Number of neighbors
    random_state=42
)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Before: {sum(y_train==0)} neg, {sum(y_train==1)} pos")
print(f"After:  {sum(y_resampled==0)} neg, {sum(y_resampled==1)} pos")


# SMOTE + Tomek links (cleaner boundaries)
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X_train, y_train)


# With Pipeline (recommended)
from imblearn.pipeline import Pipeline

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

# IMPORTANT: Only apply SMOTE to training data!
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)  # Test data NOT resampled
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
