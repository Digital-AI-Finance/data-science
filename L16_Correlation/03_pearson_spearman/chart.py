"""Pearson vs Spearman Correlation"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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
fig.suptitle('Pearson vs Spearman Correlation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Linear relationship - both work well
ax1 = axes[0, 0]
x = np.random.normal(0, 1, 50)
y = 0.8 * x + np.random.normal(0, 0.4, 50)

pearson_r = np.corrcoef(x, y)[0, 1]
spearman_r = stats.spearmanr(x, y)[0]

ax1.scatter(x, y, color=MLBLUE, alpha=0.6, s=60, edgecolors='black')
ax1.set_title(f'Linear: Pearson = {pearson_r:.2f}, Spearman = {spearman_r:.2f}', fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Monotonic non-linear - Spearman better
ax2 = axes[0, 1]
x = np.random.uniform(0.1, 5, 50)
y = np.log(x) + np.random.normal(0, 0.2, 50)

pearson_r = np.corrcoef(x, y)[0, 1]
spearman_r = stats.spearmanr(x, y)[0]

ax2.scatter(x, y, color=MLGREEN, alpha=0.6, s=60, edgecolors='black')
ax2.set_title(f'Monotonic: Pearson = {pearson_r:.2f}, Spearman = {spearman_r:.2f}', fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y = log(X) + noise', fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: With outliers - Spearman more robust
ax3 = axes[1, 0]
x = np.random.normal(0, 1, 48)
y = 0.8 * x + np.random.normal(0, 0.3, 48)
# Add outliers
x = np.append(x, [3, -3])
y = np.append(y, [-2, 2])  # Outliers that reduce Pearson

pearson_r = np.corrcoef(x, y)[0, 1]
spearman_r = stats.spearmanr(x, y)[0]

ax3.scatter(x[:-2], y[:-2], color=MLORANGE, alpha=0.6, s=60, edgecolors='black', label='Regular')
ax3.scatter(x[-2:], y[-2:], color=MLRED, s=100, edgecolors='black', marker='*', label='Outliers')
ax3.set_title(f'With Outliers: Pearson = {pearson_r:.2f}, Spearman = {spearman_r:.2f}', fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Y', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Summary comparison
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'When to Use Each', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

comparisons = [
    ('Pearson', ['Measures LINEAR relationship', 'Assumes normal distribution',
                 'Sensitive to outliers', 'Use: regression, portfolio optimization'], MLBLUE),
    ('Spearman', ['Measures MONOTONIC relationship', 'No distribution assumption',
                  'Robust to outliers', 'Use: rankings, ordinal data'], MLGREEN),
]

y = 0.75
for name, points, color in comparisons:
    ax4.text(0.1, y, name + ':', fontsize=12, fontweight='bold', color=color, transform=ax4.transAxes)
    for i, point in enumerate(points):
        ax4.text(0.15, y - 0.07 - i * 0.06, f'- {point}', fontsize=9, transform=ax4.transAxes)
    y -= 0.4

ax4.text(0.5, 0.08, 'df.corr(method="pearson")  vs  df.corr(method="spearman")',
         ha='center', fontsize=10, family='monospace', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
