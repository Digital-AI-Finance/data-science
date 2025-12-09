"""CAPM Beta - Finance application of linear regression"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
fig.suptitle('CAPM Beta Estimation Using Linear Regression', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate realistic stock and market returns
n = 252  # Trading days
market_returns = np.random.normal(0.05, 1.2, n)  # Market daily returns

# Create different stocks with different betas
stocks = {
    'AAPL': {'beta': 1.2, 'alpha': 0.02},
    'JNJ': {'beta': 0.7, 'alpha': 0.01},
    'TSLA': {'beta': 1.8, 'alpha': 0.03}
}

for stock, params in stocks.items():
    stocks[stock]['returns'] = params['alpha'] + params['beta'] * market_returns + np.random.normal(0, 1.5, n)

# Plot 1: Single stock regression (AAPL)
ax1 = axes[0, 0]

stock_ret = stocks['AAPL']['returns']
ax1.scatter(market_returns, stock_ret, c=MLBLUE, s=30, alpha=0.5, edgecolors='none')

# Fit regression
beta = np.sum((market_returns - market_returns.mean()) * (stock_ret - stock_ret.mean())) / np.sum((market_returns - market_returns.mean())**2)
alpha = stock_ret.mean() - beta * market_returns.mean()

x_line = np.linspace(market_returns.min(), market_returns.max(), 100)
ax1.plot(x_line, alpha + beta * x_line, color=MLGREEN, linewidth=2.5,
         label=f'AAPL: beta = {beta:.2f}')

ax1.axhline(0, color='gray', linewidth=1, linestyle='--')
ax1.axvline(0, color='gray', linewidth=1, linestyle='--')

ax1.set_title('AAPL vs Market Returns', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Market Return (%)', fontsize=10)
ax1.set_ylabel('AAPL Return (%)', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Add interpretation
ax1.text(0.05, 0.95, f'Beta = {beta:.2f}\nMeaning: AAPL moves\n{beta:.0%} as much as market',
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 2: Compare multiple stocks
ax2 = axes[0, 1]

colors = {'AAPL': MLBLUE, 'JNJ': MLGREEN, 'TSLA': MLORANGE}
betas_calc = {}

for stock, color in colors.items():
    ret = stocks[stock]['returns']
    beta = np.sum((market_returns - market_returns.mean()) * (ret - ret.mean())) / np.sum((market_returns - market_returns.mean())**2)
    alpha = ret.mean() - beta * market_returns.mean()
    betas_calc[stock] = beta

    ax2.plot(x_line, alpha + beta * x_line, color=color, linewidth=2.5,
             label=f'{stock}: beta = {beta:.2f}')

ax2.axhline(0, color='gray', linewidth=1, linestyle='--')
ax2.axvline(0, color='gray', linewidth=1, linestyle='--')

ax2.set_title('Multiple Stocks: Different Betas', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Market Return (%)', fontsize=10)
ax2.set_ylabel('Stock Return (%)', fontsize=10)
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(alpha=0.3)

# Plot 3: Beta interpretation
ax3 = axes[1, 0]

beta_values = list(betas_calc.values())
stock_names = list(betas_calc.keys())

bars = ax3.bar(stock_names, beta_values, color=[colors[s] for s in stock_names],
               edgecolor='black', linewidth=0.5)

ax3.axhline(1, color=MLRED, linestyle='--', linewidth=2, label='Market Beta = 1')
ax3.axhline(0, color='gray', linewidth=1)

# Add value labels
for bar, val in zip(bars, beta_values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
             ha='center', fontsize=11, fontweight='bold')

# Add interpretation zones
ax3.axhspan(0, 1, alpha=0.1, color=MLGREEN, label='Low volatility')
ax3.axhspan(1, 2, alpha=0.1, color=MLRED, label='High volatility')

ax3.set_title('Beta Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Beta', fontsize=10)
ax3.legend(fontsize=8, loc='upper right')
ax3.set_ylim(0, 2.2)
ax3.grid(alpha=0.3, axis='y')

# Plot 4: CAPM formula and interpretation
ax4 = axes[1, 1]
ax4.axis('off')

capm_text = '''
CAPM REGRESSION

Model: R_stock = alpha + beta * R_market + epsilon

Where:
- R_stock: Stock excess return (above risk-free)
- R_market: Market excess return
- alpha: Stock-specific return (skill)
- beta: Systematic risk exposure
- epsilon: Random error

BETA INTERPRETATION:

beta > 1: Aggressive stock
  - Amplifies market moves
  - Higher risk, higher potential return
  - Example: TSLA (beta = 1.8)

beta = 1: Market-tracking
  - Moves with market
  - Example: Index funds

beta < 1: Defensive stock
  - Dampens market moves
  - Lower risk, lower return
  - Example: JNJ (beta = 0.7)

beta < 0: Hedge (rare)
  - Moves opposite to market
  - Example: Gold, VIX

ALPHA INTERPRETATION:
- alpha > 0: Outperforms (after risk adjustment)
- alpha = 0: Fair priced
- alpha < 0: Underperforms
'''

ax4.text(0.02, 0.98, capm_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('CAPM Formula & Interpretation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
