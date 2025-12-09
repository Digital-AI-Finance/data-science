"""Finance with Seaborn - Comprehensive financial visualizations"""
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
fig.suptitle('Financial Analysis with Seaborn', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Return distribution comparison
ax1 = axes[0, 0]

# Different asset class returns
asset_returns = pd.DataFrame({
    'Return': np.concatenate([
        np.random.normal(0.8, 2, 252),    # Stocks
        np.random.normal(0.2, 0.5, 252),  # Bonds
        np.random.normal(0.4, 4, 252),    # Commodities
        np.random.normal(1.0, 8, 252)     # Crypto
    ]),
    'Asset': ['Stocks'] * 252 + ['Bonds'] * 252 + ['Commodities'] * 252 + ['Crypto'] * 252
})

sns.violinplot(data=asset_returns, x='Asset', y='Return', ax=ax1,
               palette=[MLBLUE, MLGREEN, MLORANGE, MLRED],
               inner='box', linewidth=1)

ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_title('Daily Return Distributions by Asset Class', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Asset Class', fontsize=10)
ax1.set_ylabel('Daily Return (%)', fontsize=10)

# Plot 2: Factor exposure heatmap
ax2 = axes[0, 1]

portfolios = ['Conservative', 'Balanced', 'Growth', 'Aggressive']
factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality']

exposures = pd.DataFrame({
    'Conservative': [0.3, 0.1, 0.3, 0.0, 0.4],
    'Balanced': [0.6, 0.2, 0.2, 0.1, 0.2],
    'Growth': [1.0, 0.3, -0.2, 0.4, 0.1],
    'Aggressive': [1.3, 0.5, -0.3, 0.6, -0.1]
}, index=factors)

sns.heatmap(exposures, ax=ax2, annot=True, fmt='.1f', cmap='RdYlBu_r',
            center=0, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Beta'})

ax2.set_title('Portfolio Factor Exposures', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Portfolio', fontsize=10)
ax2.set_ylabel('Factor', fontsize=10)

# Plot 3: Risk-adjusted returns scatter
ax3 = axes[1, 0]

# Generate fund data
n_funds = 50
fund_data = pd.DataFrame({
    'Volatility': np.random.uniform(5, 30, n_funds),
    'Return': np.random.uniform(-5, 25, n_funds),
    'Sharpe': np.random.uniform(-0.5, 2, n_funds),
    'Category': np.random.choice(['Equity', 'Fixed Income', 'Multi-Asset'], n_funds)
})

sns.scatterplot(data=fund_data, x='Volatility', y='Return', hue='Category',
                size='Sharpe', sizes=(50, 300), ax=ax3,
                palette={'Equity': MLBLUE, 'Fixed Income': MLGREEN, 'Multi-Asset': MLORANGE},
                alpha=0.7, edgecolor='black', linewidth=0.5)

# Add Sharpe = 1 line (assuming rf = 0)
x_line = np.linspace(0, 35, 100)
ax3.plot(x_line, x_line * 1, '--', color='gray', linewidth=1, label='Sharpe = 1')

ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax3.set_title('Risk-Return: Size = Sharpe Ratio', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Volatility (%)', fontsize=10)
ax3.set_ylabel('Annual Return (%)', fontsize=10)
ax3.legend(fontsize=8, loc='upper left')

# Plot 4: Sector performance boxplot with swarm
ax4 = axes[1, 1]

# Monthly returns by sector
sectors = ['Tech', 'Finance', 'Health', 'Energy', 'Consumer']
sector_returns = pd.DataFrame({
    'Sector': np.repeat(sectors, 36),  # 36 months of data
    'Monthly_Return': np.concatenate([
        np.random.normal(1.5, 5, 36),   # Tech
        np.random.normal(0.8, 3, 36),   # Finance
        np.random.normal(1.0, 4, 36),   # Health
        np.random.normal(0.5, 6, 36),   # Energy
        np.random.normal(0.7, 3, 36)    # Consumer
    ])
})

sns.boxplot(data=sector_returns, x='Sector', y='Monthly_Return', ax=ax4,
            palette=[MLBLUE, MLGREEN, MLORANGE, MLRED, MLPURPLE],
            width=0.6, linewidth=1.5)

# Overlay individual points
sns.stripplot(data=sector_returns, x='Sector', y='Monthly_Return', ax=ax4,
              color='black', alpha=0.3, size=4, jitter=0.2)

ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
ax4.set_title('Monthly Returns by Sector (3 Years)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Sector', fontsize=10)
ax4.set_ylabel('Monthly Return (%)', fontsize=10)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
