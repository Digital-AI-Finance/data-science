"""CAPM Beta - Finance application of linear regression"""
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

# Generate realistic market returns (252 trading days)
n = 252
market_returns = np.random.normal(0.05, 1.2, n)

# Generate AAPL returns with beta = 1.2
beta_true = 1.2
alpha_true = 0.02
stock_returns = alpha_true + beta_true * market_returns + np.random.normal(0, 1.5, n)

# Fit regression
beta = np.sum((market_returns - market_returns.mean()) * (stock_returns - stock_returns.mean())) / np.sum((market_returns - market_returns.mean())**2)
alpha = stock_returns.mean() - beta * market_returns.mean()

# Plot scatter
ax.scatter(market_returns, stock_returns, c=MLBLUE, s=30, alpha=0.5, edgecolors='none', label='Daily returns')

# Plot regression line
x_line = np.linspace(market_returns.min(), market_returns.max(), 100)
ax.plot(x_line, alpha + beta * x_line, color=MLGREEN, linewidth=2.5,
        label=f'AAPL: Beta = {beta:.2f}')

# Add reference lines
ax.axhline(0, color='gray', linewidth=1, linestyle='--')
ax.axvline(0, color='gray', linewidth=1, linestyle='--')

ax.set_title('CAPM Beta Estimation: Stock vs Market Returns', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Market Return (%)', fontsize=10)
ax.set_ylabel('AAPL Return (%)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

# Add interpretation box
interpretation = (f'Beta = {beta:.2f}\n'
                  f'Meaning: AAPL moves\n'
                  f'{abs(beta):.0%} as much as market\n'
                  f'Beta > 1: Aggressive stock')
ax.text(0.98, 0.02, interpretation, transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
