"""Time Series CV - Walk-forward validation"""
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
fig.suptitle('Time Series Cross-Validation: Walk-Forward Method', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why standard CV fails for time series
ax1 = axes[0, 0]
ax1.axis('off')

# Show standard K-fold problem
n_samples = 100
k = 5
fold_size = n_samples // k

# Standard CV mixes past and future
for fold in range(k):
    for i in range(k):
        y_pos = k - fold - 1
        x_start = i * fold_size
        x_end = (i + 1) * fold_size

        if i == fold:
            color = MLRED
        else:
            color = MLBLUE

        rect = mpatches.Rectangle((x_start, y_pos), fold_size - 1, 0.8,
                                   color=color, alpha=0.7, edgecolor='black')
        ax1.add_patch(rect)

ax1.text(50, -0.8, 'Standard CV: Future data trains model predicting past!', fontsize=10,
         ha='center', color=MLRED, fontweight='bold')

ax1.set_xlim(-5, n_samples + 5)
ax1.set_ylim(-1.5, k + 0.5)
ax1.set_title('Problem: Standard CV Leaks Future Information', fontsize=11, fontweight='bold', color=MLRED)

ax1.arrow(80, 2.4, -30, 0, head_width=0.3, head_length=2, fc=MLRED, ec=MLRED)
ax1.text(60, 2.8, 'Future trains\npast!', fontsize=9, color=MLRED, ha='center')

# Plot 2: Walk-forward CV (correct approach)
ax2 = axes[0, 1]
ax2.axis('off')

# Expanding window walk-forward
train_sizes = [20, 30, 40, 50, 60]
test_size = 10

for i, train_end in enumerate(train_sizes):
    y_pos = len(train_sizes) - i - 1

    # Training data (blue)
    train_rect = mpatches.Rectangle((0, y_pos), train_end, 0.8,
                                     color=MLBLUE, alpha=0.7, edgecolor='black')
    ax2.add_patch(train_rect)

    # Test data (orange)
    test_rect = mpatches.Rectangle((train_end, y_pos), test_size, 0.8,
                                    color=MLORANGE, alpha=0.7, edgecolor='black')
    ax2.add_patch(test_rect)

    # Unused future data (gray)
    unused_rect = mpatches.Rectangle((train_end + test_size, y_pos), 90 - train_end - test_size, 0.8,
                                      color='lightgray', alpha=0.3, edgecolor='black')
    ax2.add_patch(unused_rect)

    ax2.text(-5, y_pos + 0.4, f'Split {i+1}', fontsize=9, va='center')

ax2.set_xlim(-15, 95)
ax2.set_ylim(-0.5, len(train_sizes) + 0.5)
ax2.set_title('Solution: Walk-Forward CV (Expanding Window)', fontsize=11, fontweight='bold', color=MLGREEN)

# Add timeline
ax2.arrow(0, -0.3, 90, 0, head_width=0.15, head_length=2, fc='black', ec='black')
ax2.text(45, -0.6, 'Time', fontsize=10, ha='center')

# Legend
train_patch = mpatches.Patch(color=MLBLUE, label='Training')
test_patch = mpatches.Patch(color=MLORANGE, label='Test')
ax2.legend(handles=[train_patch, test_patch], loc='upper right', fontsize=9)

# Plot 3: Walk-forward results
ax3 = axes[1, 0]

splits = ['Split 1', 'Split 2', 'Split 3', 'Split 4', 'Split 5']
rmse_scores = [2.8, 2.5, 2.3, 2.6, 2.4]

bars = ax3.bar(splits, rmse_scores, color=MLBLUE, edgecolor='black', linewidth=0.5)
ax3.axhline(np.mean(rmse_scores), color=MLRED, linestyle='--', linewidth=2,
            label=f'Mean RMSE: {np.mean(rmse_scores):.2f}')

ax3.set_title('Walk-Forward CV Results', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('RMSE', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')

# Add std annotation
ax3.text(4, np.mean(rmse_scores) + 0.1, f'Std: {np.std(rmse_scores):.2f}',
         fontsize=10, color=MLRED)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
TIME SERIES CROSS-VALIDATION IN SKLEARN

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# Create time series splitter
tscv = TimeSeriesSplit(n_splits=5)

# Walk-forward validation
scores = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(rmse)

print(f"Mean RMSE: {np.mean(scores):.3f}")
print(f"Std RMSE:  {np.std(scores):.3f}")

# Alternative: use cross_val_score
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, cv=tscv,
    scoring='neg_root_mean_squared_error'
)
'''

ax4.text(0.02, 0.95, code, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
