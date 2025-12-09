"""Scatter Plots - Relationship visualization"""
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
fig.suptitle('Scatter Plots with matplotlib', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic scatter
ax1 = axes[0, 0]
x = np.random.normal(0, 1, 100)
y = 0.7 * x + np.random.normal(0, 0.5, 100)

ax1.scatter(x, y, c=MLBLUE, s=50, alpha=0.6, edgecolors='black')

ax1.set_title('Basic Scatter Plot', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Scatter with regression line
ax2 = axes[0, 1]
risk = np.random.uniform(5, 25, 30)  # Portfolio risk
returns = 2 + 0.4 * risk + np.random.normal(0, 2, 30)  # Expected return

ax2.scatter(risk, returns, c=MLGREEN, s=80, alpha=0.7, edgecolors='black')

# Regression line
z = np.polyfit(risk, returns, 1)
ax2.plot(np.sort(risk), np.poly1d(z)(np.sort(risk)), color=MLRED, linewidth=2.5,
         label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

ax2.set_title('Risk vs Return (with regression)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Risk (%)', fontsize=10)
ax2.set_ylabel('Expected Return (%)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Size and color encoding
ax3 = axes[1, 0]
n = 50
market_cap = np.random.uniform(10, 500, n)  # Billions
returns_pct = np.random.normal(5, 10, n)
volatility = np.random.uniform(10, 40, n)

scatter = ax3.scatter(volatility, returns_pct, c=market_cap, s=market_cap/2,
                      cmap='viridis', alpha=0.7, edgecolors='black')
plt.colorbar(scatter, ax=ax3, label='Market Cap ($B)')

ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_title('Multi-dimensional: Size = MarketCap, Color = MarketCap', fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Volatility (%)', fontsize=10)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: Grouped scatter
ax4 = axes[1, 1]
# Three asset classes
for name, color, mean_ret, mean_risk in [('Stocks', MLBLUE, 8, 18),
                                          ('Bonds', MLGREEN, 3, 6),
                                          ('Commodities', MLORANGE, 5, 22)]:
    risk = np.random.normal(mean_risk, 3, 20)
    returns = np.random.normal(mean_ret, 2, 20)
    ax4.scatter(risk, returns, c=color, s=60, alpha=0.7, edgecolors='black', label=name)

ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
ax4.set_title('Asset Class Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Risk (%)', fontsize=10)
ax4.set_ylabel('Return (%)', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
