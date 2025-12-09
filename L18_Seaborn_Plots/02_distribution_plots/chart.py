"""Distribution Plots - Visualizing data distributions with seaborn"""
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
fig.suptitle('Seaborn Distribution Plots', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: histplot with multiple options
ax1 = axes[0, 0]
returns = np.random.normal(0, 2, 1000)

sns.histplot(returns, bins=40, kde=True, ax=ax1, color=MLBLUE, alpha=0.6,
             stat='density', edgecolor='white', linewidth=0.5,
             line_kws={'linewidth': 2, 'color': MLRED})

ax1.set_title('histplot: Histogram + KDE', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Daily Return (%)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)

# Add annotation
ax1.text(0.95, 0.95, 'stat="density"\nkde=True',
         transform=ax1.transAxes, ha='right', va='top',
         fontsize=9, bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 2: kdeplot - kernel density estimation
ax2 = axes[0, 1]

# Multiple distributions
tech_returns = np.random.normal(0.5, 2.5, 500)
bond_returns = np.random.normal(0.1, 0.8, 500)
commodity_returns = np.random.normal(0.2, 3.5, 500)

sns.kdeplot(tech_returns, ax=ax2, color=MLBLUE, linewidth=2.5, label='Tech Stocks')
sns.kdeplot(bond_returns, ax=ax2, color=MLGREEN, linewidth=2.5, label='Bonds')
sns.kdeplot(commodity_returns, ax=ax2, color=MLORANGE, linewidth=2.5, label='Commodities')

ax2.set_title('kdeplot: Compare Distributions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Return (%)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=9)

# Plot 3: rugplot and displot features
ax3 = axes[1, 0]

sample_returns = np.random.normal(1, 2, 100)

sns.histplot(sample_returns, kde=True, ax=ax3, color=MLBLUE, alpha=0.5,
             edgecolor='white', linewidth=0.5)
sns.rugplot(sample_returns, ax=ax3, color=MLRED, alpha=0.5, height=0.05)

ax3.set_title('histplot + rugplot: Show Individual Points', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Return (%)', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)

ax3.text(0.95, 0.95, 'Rug marks show\neach observation',
         transform=ax3.transAxes, ha='right', va='top',
         fontsize=9, bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 4: ecdfplot - empirical cumulative distribution
ax4 = axes[1, 1]

# Create asset returns
np.random.seed(42)
df = pd.DataFrame({
    'Return': np.concatenate([
        np.random.normal(0.5, 2, 200),
        np.random.normal(0.3, 1.5, 200),
        np.random.normal(0.2, 3, 200)
    ]),
    'Asset': ['Stock'] * 200 + ['Bond'] * 200 + ['Crypto'] * 200
})

for asset, color in zip(['Stock', 'Bond', 'Crypto'], [MLBLUE, MLGREEN, MLORANGE]):
    data = df[df['Asset'] == asset]['Return']
    sns.ecdfplot(data, ax=ax4, color=color, linewidth=2.5, label=asset)

ax4.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax4.text(ax4.get_xlim()[1] * 0.95, 0.52, 'Median line', fontsize=8, color='gray')

ax4.set_title('ecdfplot: Cumulative Distribution', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Return (%)', fontsize=10)
ax4.set_ylabel('Cumulative Probability', fontsize=10)
ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
