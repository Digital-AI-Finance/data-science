"""PCA for Factor Extraction - Finance Application"""
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
fig.suptitle('PCA for Factor Extraction in Finance', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate synthetic stock return data
n_days = 252
n_stocks = 20
tickers = [f'STK{i:02d}' for i in range(n_stocks)]

# Create correlated returns (market + sector factors)
market = np.random.randn(n_days) * 0.01

# Tech stocks (0-7)
tech_factor = np.random.randn(n_days) * 0.008
tech = np.column_stack([market + tech_factor + np.random.randn(n_days) * 0.005 for _ in range(8)])

# Financial stocks (8-13)
fin_factor = np.random.randn(n_days) * 0.007
fin = np.column_stack([market + fin_factor + np.random.randn(n_days) * 0.005 for _ in range(6)])

# Industrial stocks (14-19)
ind_factor = np.random.randn(n_days) * 0.006
ind = np.column_stack([market + ind_factor + np.random.randn(n_days) * 0.005 for _ in range(6)])

returns = np.hstack([tech, fin, ind])

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

pca = PCA()
pca.fit(returns_scaled)

# Plot 1: Explained variance
ax1 = axes[0, 0]

n_components = len(pca.explained_variance_ratio_)
cumsum = np.cumsum(pca.explained_variance_ratio_) * 100

ax1.bar(range(1, n_components+1), pca.explained_variance_ratio_ * 100,
        color=MLBLUE, edgecolor='black', alpha=0.7, label='Individual')
ax1.plot(range(1, n_components+1), cumsum, 'o-', color=MLRED,
         linewidth=2, markersize=6, label='Cumulative')

ax1.axhline(90, color=MLGREEN, linestyle='--', alpha=0.7)

ax1.set_title('Factor Extraction from 20 Stocks', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Principal Component (Factor)')
ax1.set_ylabel('Explained Variance (%)')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, axis='y')

# Annotate
pc1_var = pca.explained_variance_ratio_[0] * 100
ax1.text(1.5, pc1_var + 2, f'PC1: {pc1_var:.1f}%\n(Market Factor)',
         fontsize=9, color=MLBLUE, fontweight='bold')

# Plot 2: Factor loadings
ax2 = axes[0, 1]

# Show loadings for first 3 factors
loadings = pca.components_[:3]

im = ax2.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-0.4, vmax=0.4)

ax2.set_xticks(range(n_stocks))
ax2.set_xticklabels([f'{i}' for i in range(n_stocks)], fontsize=7)
ax2.set_yticks(range(3))
ax2.set_yticklabels(['Factor 1\n(Market)', 'Factor 2\n(Tech vs Fin)', 'Factor 3\n(Industry)'],
                    fontsize=8)

# Add sector dividers
ax2.axvline(7.5, color='black', linewidth=2)
ax2.axvline(13.5, color='black', linewidth=2)

ax2.text(3.5, -0.7, 'Tech', fontsize=9, ha='center', fontweight='bold', color=MLBLUE)
ax2.text(10.5, -0.7, 'Finance', fontsize=9, ha='center', fontweight='bold', color=MLGREEN)
ax2.text(16.5, -0.7, 'Industrial', fontsize=9, ha='center', fontweight='bold', color=MLORANGE)

ax2.set_title('Factor Loadings by Stock', fontsize=11, fontweight='bold', color=MLPURPLE)
plt.colorbar(im, ax=ax2, shrink=0.7, label='Loading')

# Plot 3: Finance application
ax3 = axes[1, 0]
ax3.axis('off')

application = '''
PCA FOR FACTOR EXTRACTION IN FINANCE

USE CASE: Finding common factors in stock returns


WHAT PCA REVEALS:
-----------------
PC1 (40-60%): Market factor
  - All loadings positive
  - Captures overall market movement
  - Similar to S&P 500

PC2-5 (20-30%): Sector/Style factors
  - Some positive, some negative
  - Tech vs. Value, Growth vs. Defensive
  - Captures sector rotation

Remaining PCs: Idiosyncratic (stock-specific)


APPLICATIONS:
-------------
1. RISK DECOMPOSITION
   How much risk is market vs sector vs stock?

2. FACTOR INVESTING
   Build factor-based portfolios

3. ANOMALY DETECTION
   Stocks that don't follow factors = unusual

4. PORTFOLIO CONSTRUCTION
   Diversify across factors, not just stocks


STATISTICAL ARBITRAGE:
----------------------
If PC1 = market factor, then:
Residuals = returns unexplained by market

Long stocks with positive residuals
Short stocks with negative residuals
(Mean-reversion strategy)
'''

ax3.text(0.02, 0.98, application, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Finance Applications', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
PCA FACTOR EXTRACTION CODE

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load stock returns
prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()

# Standardize
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

# Extract factors
pca = PCA(n_components=5)
factors = pca.fit_transform(returns_scaled)

# Factor returns as DataFrame
factor_returns = pd.DataFrame(
    factors,
    index=returns.index,
    columns=['Market', 'Sector', 'Style', 'F4', 'F5']
)


# Variance explained
print("Variance explained:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  Factor {i+1}: {var*100:.1f}%")


# Loadings (factor exposures)
loadings = pd.DataFrame(
    pca.components_.T,
    index=returns.columns,
    columns=factor_returns.columns
)
print("\\nFactor loadings:")
print(loadings.round(3))


# Reconstruct returns (for residuals)
returns_reconstructed = pca.inverse_transform(factors)
residuals = returns_scaled - returns_reconstructed
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
