"""Portfolio Correlation - Diversification benefits"""
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
fig.suptitle('Portfolio Correlation and Diversification', fontsize=14,
             fontweight='bold', color=MLPURPLE)

# Generate asset returns
n = 252
base = np.random.normal(0.0004, 0.01, n)

stocks = base + np.random.normal(0, 0.005, n)
bonds = -0.3 * base + np.random.normal(0.0002, 0.003, n)
gold = 0.1 * base + np.random.normal(0.0001, 0.007, n)

df = pd.DataFrame({
    'Stocks': stocks,
    'Bonds': bonds,
    'Gold': gold
})

# Plot 1: Efficient frontier concept
ax1 = axes[0, 0]

# Calculate portfolio risk-return for different stock-bond mixes
weights_range = np.linspace(0, 1, 50)
portfolio_returns = []
portfolio_risks = []

stock_ret, bond_ret = df['Stocks'].mean() * 252, df['Bonds'].mean() * 252
stock_vol, bond_vol = df['Stocks'].std() * np.sqrt(252), df['Bonds'].std() * np.sqrt(252)
corr = df['Stocks'].corr(df['Bonds'])

for w in weights_range:
    port_ret = w * stock_ret + (1-w) * bond_ret
    port_var = w**2 * stock_vol**2 + (1-w)**2 * bond_vol**2 + 2*w*(1-w)*stock_vol*bond_vol*corr
    portfolio_returns.append(port_ret * 100)
    portfolio_risks.append(np.sqrt(port_var) * 100)

ax1.plot(portfolio_risks, portfolio_returns, color=MLBLUE, linewidth=2.5)
ax1.scatter([stock_vol * 100], [stock_ret * 100], color=MLRED, s=100, zorder=5, label='100% Stocks')
ax1.scatter([bond_vol * 100], [bond_ret * 100], color=MLGREEN, s=100, zorder=5, label='100% Bonds')

# Mark minimum variance portfolio
min_var_idx = np.argmin(portfolio_risks)
ax1.scatter([portfolio_risks[min_var_idx]], [portfolio_returns[min_var_idx]],
            color=MLPURPLE, s=150, marker='*', zorder=5, label='Min Variance')

ax1.set_title('Efficient Frontier (Stocks + Bonds)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Portfolio Risk (%)', fontsize=10)
ax1.set_ylabel('Expected Return (%)', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Correlation matrix
ax2 = axes[0, 1]
corr_matrix = df.corr()
im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax2, shrink=0.8)

ax2.set_xticks(range(3))
ax2.set_yticks(range(3))
ax2.set_xticklabels(['Stocks', 'Bonds', 'Gold'])
ax2.set_yticklabels(['Stocks', 'Bonds', 'Gold'])

for i in range(3):
    for j in range(3):
        ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center',
                 fontsize=12, color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

ax2.set_title('Asset Correlation Matrix', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Risk reduction with more assets
ax3 = axes[1, 0]

# Simulate portfolios with 1 to 20 uncorrelated assets
max_assets = 20
avg_risks = []
asset_counts = range(1, max_assets + 1)

for n_assets in asset_counts:
    # Generate uncorrelated returns
    all_returns = np.random.normal(0.0005, 0.02, (252, n_assets))
    # Equal weighted portfolio
    portfolio = all_returns.mean(axis=1)
    avg_risks.append(portfolio.std() * np.sqrt(252) * 100)

ax3.plot(asset_counts, avg_risks, 'o-', color=MLBLUE, linewidth=2, markersize=6)
ax3.axhline(avg_risks[-1], color=MLRED, linestyle='--', label=f'Systematic risk: {avg_risks[-1]:.1f}%')

ax3.set_title('Diversification: Risk vs Number of Assets', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Number of Assets', fontsize=10)
ax3.set_ylabel('Portfolio Risk (%)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

ax3.annotate('Diversifiable\n(idiosyncratic)', xy=(5, avg_risks[4]), xytext=(8, avg_risks[4] + 5),
             fontsize=9, arrowprops=dict(arrowstyle='->', color=MLGREEN))
ax3.annotate('Non-diversifiable\n(systematic)', xy=(18, avg_risks[-1] + 0.5), fontsize=9, color=MLRED)

# Plot 4: Key insights
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Diversification Insights', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

insights = [
    ('Lower Correlation = Better Diversification', 'Negative correlation is ideal but rare'),
    ('Diminishing Returns', '~15-20 stocks capture most diversification'),
    ('Systematic Risk Remains', 'Cannot diversify away market risk'),
    ('Correlations Rise in Crises', 'Diversification fails when needed most'),
]

y = 0.75
for title, desc in insights:
    ax4.text(0.1, y, title, fontsize=10, fontweight='bold', color=MLBLUE, transform=ax4.transAxes)
    ax4.text(0.1, y - 0.07, desc, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.17

ax4.text(0.5, 0.1, 'Portfolio_Var = sum(wi*wj*cov_ij)',
         ha='center', fontsize=10, family='monospace', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
