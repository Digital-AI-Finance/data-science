"""Correlation Scatter - Visualizing relationships"""
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
fig.suptitle('Correlation: Visualizing Relationships', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Strong positive correlation
ax1 = axes[0, 0]
x = np.random.normal(0, 1, 100)
y = 0.9 * x + np.random.normal(0, 0.3, 100)
r = np.corrcoef(x, y)[0, 1]

ax1.scatter(x, y, color=MLBLUE, alpha=0.6, s=50, edgecolors='black')
z = np.polyfit(x, y, 1)
ax1.plot(np.sort(x), np.poly1d(z)(np.sort(x)), color=MLRED, linewidth=2.5, label='Best fit')

ax1.set_title(f'Strong Positive: r = {r:.2f}', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Weak negative correlation
ax2 = axes[0, 1]
x = np.random.normal(0, 1, 100)
y = -0.3 * x + np.random.normal(0, 1, 100)
r = np.corrcoef(x, y)[0, 1]

ax2.scatter(x, y, color=MLORANGE, alpha=0.6, s=50, edgecolors='black')
z = np.polyfit(x, y, 1)
ax2.plot(np.sort(x), np.poly1d(z)(np.sort(x)), color=MLRED, linewidth=2.5)

ax2.set_title(f'Weak Negative: r = {r:.2f}', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: Finance example - Two stocks
ax3 = axes[1, 0]
n = 252
market = np.random.normal(0.0004, 0.01, n)
stock = 1.2 * market + np.random.normal(0, 0.005, n)  # Beta = 1.2

ax3.scatter(market * 100, stock * 100, color=MLGREEN, alpha=0.5, s=30, edgecolors='black')

# Regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(market, stock)
ax3.plot(np.sort(market) * 100, (slope * np.sort(market) + intercept) * 100,
         color=MLRED, linewidth=2.5, label=f'Beta = {slope:.2f}')

ax3.set_title(f'Stock vs Market Returns (r = {r_value:.2f})', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Market Return (%)', fontsize=10)
ax3.set_ylabel('Stock Return (%)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.axhline(0, color='gray', linewidth=1)
ax3.axvline(0, color='gray', linewidth=1)

# Plot 4: No correlation
ax4 = axes[1, 1]
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
r = np.corrcoef(x, y)[0, 1]

ax4.scatter(x, y, color=MLRED, alpha=0.6, s=50, edgecolors='black')
ax4.axhline(np.mean(y), color=MLORANGE, linestyle='--', linewidth=1.5, label='Mean Y')
ax4.axvline(np.mean(x), color=MLBLUE, linestyle='--', linewidth=1.5, label='Mean X')

ax4.set_title(f'No Correlation: r = {r:.2f}', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('X', fontsize=10)
ax4.set_ylabel('Y', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
