"""Heatmaps - Visualizing matrix data with seaborn"""
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
fig.suptitle('Seaborn Heatmaps', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Correlation heatmap
ax1 = axes[0, 0]

# Create correlated asset returns
n_days = 252
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'XOM']

# Generate correlated returns
cov = np.random.uniform(0.3, 0.8, (8, 8))
cov = (cov + cov.T) / 2  # Make symmetric
np.fill_diagonal(cov, 1)

# Create correlation matrix
returns_df = pd.DataFrame(
    np.random.multivariate_normal(np.zeros(8), cov, n_days),
    columns=assets
)
corr_matrix = returns_df.corr()

sns.heatmap(corr_matrix, ax=ax1, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})

ax1.set_title('Correlation Matrix: Asset Returns', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Monthly returns heatmap
ax2 = axes[0, 1]

# Create monthly returns data
years = [2020, 2021, 2022, 2023, 2024]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

monthly_returns = pd.DataFrame(
    np.random.normal(0.8, 4, (len(years), 12)),
    index=years,
    columns=months
)

sns.heatmap(monthly_returns, ax=ax2, annot=True, fmt='.1f', cmap='RdYlGn',
            center=0, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Return (%)'})

ax2.set_title('Monthly Returns by Year (%)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Month', fontsize=10)
ax2.set_ylabel('Year', fontsize=10)

# Plot 3: Clustered heatmap (manual clustering effect)
ax3 = axes[1, 0]

# Sector exposure matrix
sectors = ['Tech', 'Finance', 'Health', 'Energy', 'Consumer']
portfolios = ['Growth', 'Value', 'Balanced', 'Income', 'Aggressive']

# Create sector weights
weights = np.array([
    [0.40, 0.10, 0.20, 0.10, 0.20],  # Growth - tech heavy
    [0.10, 0.35, 0.15, 0.25, 0.15],  # Value - finance/energy
    [0.20, 0.20, 0.20, 0.20, 0.20],  # Balanced
    [0.05, 0.30, 0.15, 0.35, 0.15],  # Income - finance/energy
    [0.50, 0.05, 0.25, 0.05, 0.15],  # Aggressive - tech heavy
])

weights_df = pd.DataFrame(weights, index=portfolios, columns=sectors)

sns.heatmap(weights_df, ax=ax3, annot=True, fmt='.0%', cmap='Blues',
            linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Weight'})

ax3.set_title('Portfolio Sector Allocation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Sector', fontsize=10)
ax3.set_ylabel('Portfolio Strategy', fontsize=10)

# Plot 4: Diverging heatmap for performance
ax4 = axes[1, 1]

# Factor performance by regime
factors = ['Momentum', 'Value', 'Size', 'Quality', 'Volatility']
regimes = ['Bull', 'Bear', 'High Vol', 'Low Vol', 'Recovery']

factor_perf = pd.DataFrame({
    'Bull': [8.5, 3.2, 5.1, 4.3, -2.1],
    'Bear': [-5.2, 2.8, -3.1, 6.2, 4.5],
    'High Vol': [2.1, -1.5, 0.8, 3.2, -5.8],
    'Low Vol': [4.3, 4.1, 2.9, 2.1, 1.2],
    'Recovery': [12.5, 5.2, 8.3, 3.1, -3.5]
}, index=factors)

sns.heatmap(factor_perf, ax=ax4, annot=True, fmt='.1f', cmap='RdYlGn',
            center=0, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Return (%)'})

ax4.set_title('Factor Performance by Market Regime', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Market Regime', fontsize=10)
ax4.set_ylabel('Factor', fontsize=10)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
