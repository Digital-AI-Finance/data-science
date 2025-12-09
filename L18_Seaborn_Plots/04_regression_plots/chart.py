"""Regression Plots - Statistical relationships with seaborn"""
import matplotlib.pyplot as plt
import seaborn as sns
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
fig.suptitle('Seaborn Regression Plots', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: regplot - basic linear regression
ax1 = axes[0, 0]

# CAPM-like relationship: stock return vs market return
n = 100
market_return = np.random.normal(0.5, 2, n)
stock_return = 0.2 + 1.2 * market_return + np.random.normal(0, 1.5, n)  # Beta = 1.2

sns.regplot(x=market_return, y=stock_return, ax=ax1,
            color=MLBLUE, scatter_kws={'alpha': 0.6, 's': 40},
            line_kws={'color': MLRED, 'linewidth': 2})

ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

ax1.set_title('regplot: Stock vs Market (Beta Estimation)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Market Return (%)', fontsize=10)
ax1.set_ylabel('Stock Return (%)', fontsize=10)

# Add beta annotation
ax1.text(0.05, 0.95, 'Beta = 1.2\n(slope of line)',
         transform=ax1.transAxes, ha='left', va='top',
         fontsize=9, bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 2: regplot with polynomial
ax2 = axes[0, 1]

# Non-linear relationship: volatility smile
strike_pct = np.linspace(-30, 30, 50)  # % from ATM
implied_vol = 20 + 0.01 * strike_pct**2 + np.random.normal(0, 1, 50)

sns.regplot(x=strike_pct, y=implied_vol, ax=ax2,
            color=MLBLUE, scatter_kws={'alpha': 0.6, 's': 40},
            line_kws={'color': MLRED, 'linewidth': 2},
            order=2)  # Polynomial degree 2

ax2.set_title('regplot order=2: Volatility Smile', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Strike (% from ATM)', fontsize=10)
ax2.set_ylabel('Implied Volatility (%)', fontsize=10)

# Plot 3: residplot - regression diagnostics
ax3 = axes[1, 0]

# Create data with non-constant variance (heteroscedasticity)
x = np.linspace(1, 10, 100)
y = 2 + 1.5 * x + np.random.normal(0, x * 0.5, 100)  # Variance increases with x

sns.residplot(x=x, y=y, ax=ax3, color=MLBLUE,
              scatter_kws={'alpha': 0.6, 's': 40},
              lowess=True, line_kws={'color': MLRED, 'linewidth': 2})

ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_title('residplot: Check Regression Assumptions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Residuals', fontsize=10)

ax3.text(0.95, 0.95, 'Fan shape = heteroscedasticity\n(non-constant variance)',
         transform=ax3.transAxes, ha='right', va='top',
         fontsize=9, bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 4: lmplot-style with hue (using regplot manually)
ax4 = axes[1, 1]

# Risk-return relationship by market cap
df = pd.DataFrame({
    'Risk': np.concatenate([
        np.random.uniform(5, 15, 30),  # Large cap
        np.random.uniform(10, 25, 30),  # Mid cap
        np.random.uniform(15, 35, 30)   # Small cap
    ]),
    'Return': np.concatenate([
        np.random.normal(8, 2, 30),
        np.random.normal(10, 3, 30),
        np.random.normal(12, 4, 30)
    ]),
    'Market Cap': ['Large'] * 30 + ['Mid'] * 30 + ['Small'] * 30
})

for cap, color in zip(['Large', 'Mid', 'Small'], [MLBLUE, MLGREEN, MLORANGE]):
    subset = df[df['Market Cap'] == cap]
    sns.regplot(x='Risk', y='Return', data=subset, ax=ax4,
                color=color, scatter_kws={'alpha': 0.6, 's': 40},
                line_kws={'linewidth': 2}, label=f'{cap} Cap')

ax4.set_title('Grouped Regression: Risk-Return by Market Cap', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Risk (Volatility %)', fontsize=10)
ax4.set_ylabel('Expected Return (%)', fontsize=10)
ax4.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
