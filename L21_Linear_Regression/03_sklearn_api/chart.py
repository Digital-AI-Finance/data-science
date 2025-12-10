"""sklearn API - Linear regression with scikit-learn"""
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

fig, ax = plt.subplots(figsize=(10, 6))

# Generate data
n = 50
X = np.random.uniform(5, 30, n).reshape(-1, 1)
y = 3 + 0.5 * X.flatten() + np.random.normal(0, 2, n)

# Fit model (simulating sklearn)
slope = np.sum((X.flatten() - X.mean()) * (y - y.mean())) / np.sum((X.flatten() - X.mean())**2)
intercept = y.mean() - slope * X.mean()
y_pred = intercept + slope * X.flatten()
r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

# Plot
ax.scatter(X, y, c=MLBLUE, s=60, alpha=0.7, edgecolors='black', label='Training data')
ax.plot(np.sort(X.flatten()), intercept + slope * np.sort(X.flatten()),
        color=MLGREEN, linewidth=2.5, label='model.fit(X, y)')

ax.set_title('sklearn LinearRegression: Fit Result', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Risk (%)', fontsize=10)
ax.set_ylabel('Return (%)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Add coefficient box
coef_text = (f'model.coef_[0] = {slope:.3f}\n'
             f'model.intercept_ = {intercept:.3f}\n'
             f'model.score(X, y) = {r_squared:.3f}')
ax.text(0.95, 0.05, coef_text, transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLGREEN, linewidth=1.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
