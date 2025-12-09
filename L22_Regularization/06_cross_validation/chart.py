"""Cross-Validation - Robust model evaluation"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
fig.suptitle('Cross-Validation: Reliable Model Evaluation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: K-Fold visualization
ax1 = axes[0, 0]
ax1.axis('off')

n_samples = 100
k = 5
fold_size = n_samples // k

for fold in range(k):
    for i in range(k):
        y_pos = k - fold - 1
        x_start = i * fold_size
        x_end = (i + 1) * fold_size

        if i == fold:
            # Test fold
            color = MLRED
            label = 'Test' if fold == 0 and i == 0 else None
        else:
            # Train fold
            color = MLBLUE
            label = 'Train' if fold == 0 and i == 1 else None

        rect = mpatches.Rectangle((x_start, y_pos), fold_size - 2, 0.8,
                                   color=color, alpha=0.7, edgecolor='black')
        ax1.add_patch(rect)

    # Add fold label
    ax1.text(-8, k - fold - 1 + 0.4, f'Fold {fold + 1}', fontsize=10, va='center', fontweight='bold')

ax1.set_xlim(-15, n_samples + 5)
ax1.set_ylim(-0.5, k + 0.5)
ax1.set_title('5-Fold Cross-Validation Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add legend
train_patch = mpatches.Patch(color=MLBLUE, label='Training Data')
test_patch = mpatches.Patch(color=MLRED, label='Test Data')
ax1.legend(handles=[train_patch, test_patch], loc='upper right', fontsize=9)

# Plot 2: Why CV is better than single split
ax2 = axes[0, 1]

# Single split variability
n_runs = 20
single_split_scores = 0.75 + np.random.normal(0, 0.08, n_runs)
cv_scores = 0.75 + np.random.normal(0, 0.02, n_runs)

x_single = np.ones(n_runs) + np.random.uniform(-0.1, 0.1, n_runs)
x_cv = np.ones(n_runs) * 2 + np.random.uniform(-0.1, 0.1, n_runs)

ax2.scatter(x_single, single_split_scores, c=MLRED, s=60, alpha=0.6, edgecolors='black', label='Single Split')
ax2.scatter(x_cv, cv_scores, c=MLGREEN, s=60, alpha=0.6, edgecolors='black', label='5-Fold CV')

# Add mean lines
ax2.hlines(np.mean(single_split_scores), 0.7, 1.3, colors=MLRED, linewidth=2)
ax2.hlines(np.mean(cv_scores), 1.7, 2.3, colors=MLGREEN, linewidth=2)

# Add error bars
ax2.errorbar(1, np.mean(single_split_scores), yerr=np.std(single_split_scores),
             color=MLRED, capsize=5, capthick=2, linewidth=2)
ax2.errorbar(2, np.mean(cv_scores), yerr=np.std(cv_scores),
             color=MLGREEN, capsize=5, capthick=2, linewidth=2)

ax2.set_xlim(0.5, 2.5)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Single Split\n(High Variance)', '5-Fold CV\n(Low Variance)'])
ax2.set_title('CV Reduces Evaluation Variance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('R-squared Score', fontsize=10)
ax2.grid(alpha=0.3, axis='y')

# Plot 3: CV scores for different models
ax3 = axes[1, 0]

models = ['OLS', 'Ridge\n(a=0.1)', 'Ridge\n(a=1)', 'Ridge\n(a=10)', 'Lasso\n(a=0.1)', 'Lasso\n(a=1)']
cv_means = [0.72, 0.75, 0.78, 0.74, 0.76, 0.73]
cv_stds = [0.08, 0.05, 0.04, 0.05, 0.05, 0.06]

colors = [MLBLUE if m != 0.78 else MLGREEN for m in cv_means]

bars = ax3.bar(models, cv_means, color=colors, edgecolor='black', linewidth=0.5)
ax3.errorbar(models, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5)

# Highlight best
best_idx = np.argmax(cv_means)
ax3.annotate('Best Model', xy=(best_idx, cv_means[best_idx] + cv_stds[best_idx]),
             xytext=(best_idx + 0.5, cv_means[best_idx] + 0.08), fontsize=10,
             arrowprops=dict(arrowstyle='->', color=MLGREEN), color=MLGREEN, fontweight='bold')

ax3.set_title('Model Comparison via Cross-Validation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('CV R-squared', fontsize=10)
ax3.set_ylim(0.5, 0.9)
ax3.grid(alpha=0.3, axis='y')

# Plot 4: sklearn CV code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Cross-Validation with sklearn

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Basic cross-validation
model = Ridge(alpha=1.0)
scores = cross_val_score(
    model, X, y,
    cv=5,  # 5-fold CV
    scoring='r2'
)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std:  {scores.std():.3f}")

# Cross-validation with shuffle
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

# Get predictions via cross_val_predict
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(model, X, y, cv=5)
# Each point predicted when NOT in training
'''

ax4.text(0.02, 0.95, code, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
