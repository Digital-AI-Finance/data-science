"""Finance Merge - Combining multiple financial datasets"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

# Create sample financial datasets
np.random.seed(42)

# Stock prices
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
prices = pd.DataFrame({
    'Ticker': stocks,
    'Price': [175, 380, 140, 185, 500, 880],
    'Returns_YTD': np.random.uniform(-0.1, 0.3, 6)
})

# Company info
info = pd.DataFrame({
    'Ticker': stocks[:5] + ['TSLA'],  # Different stocks
    'Sector': ['Tech', 'Tech', 'Tech', 'Consumer', 'Tech', 'Auto'],
    'MarketCap_B': [2800, 2500, 1800, 1900, 1300, 800]
})

# Merge result
merged = pd.merge(prices, info, on='Ticker', how='inner')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Financial Data Merge Example', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Before merge - two bar charts
ax1 = axes[0, 0]
x = np.arange(len(prices))
ax1.bar(x, prices['Price'], color=MLBLUE, alpha=0.7, edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(prices['Ticker'], rotation=45)
ax1.set_ylabel('Stock Price ($)', fontsize=10)
ax1.set_title('df_prices: 6 stocks', fontsize=11, fontweight='bold', color=MLBLUE)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Company info
ax2 = axes[0, 1]
x2 = np.arange(len(info))
ax2.bar(x2, info['MarketCap_B'], color=MLORANGE, alpha=0.7, edgecolor='black')
ax2.set_xticks(x2)
ax2.set_xticklabels(info['Ticker'], rotation=45)
ax2.set_ylabel('Market Cap ($B)', fontsize=10)
ax2.set_title('df_info: 6 stocks (different set)', fontsize=11, fontweight='bold', color=MLORANGE)
ax2.grid(axis='y', alpha=0.3)

# Highlight overlap
ax1.text(0.5, 0.95, 'NVDA not in df_info', transform=ax1.transAxes,
         fontsize=9, color=MLRED, style='italic', ha='left', va='top')
ax2.text(0.5, 0.95, 'TSLA not in df_prices', transform=ax2.transAxes,
         fontsize=9, color=MLRED, style='italic', ha='left', va='top')

# Plot 3: After merge - scatter plot
ax3 = axes[1, 0]
sectors = merged['Sector'].unique()
colors_sector = {'Tech': MLBLUE, 'Consumer': MLORANGE}
for sector in sectors:
    subset = merged[merged['Sector'] == sector]
    ax3.scatter(subset['MarketCap_B'], subset['Price'], s=100,
                label=sector, color=colors_sector.get(sector, MLPURPLE), alpha=0.7)
    for _, row in subset.iterrows():
        ax3.annotate(row['Ticker'], (row['MarketCap_B'], row['Price']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

ax3.set_xlabel('Market Cap ($B)', fontsize=10)
ax3.set_ylabel('Stock Price ($)', fontsize=10)
ax3.set_title('After Inner Merge: 5 matching stocks', fontsize=11, fontweight='bold', color=MLGREEN)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Returns by sector
ax4 = axes[1, 1]
sector_returns = merged.groupby('Sector')['Returns_YTD'].mean() * 100
colors_bar = [colors_sector.get(s, MLPURPLE) for s in sector_returns.index]
bars = ax4.bar(sector_returns.index, sector_returns.values, color=colors_bar, alpha=0.7, edgecolor='black')

ax4.set_ylabel('Average YTD Returns (%)', fontsize=10)
ax4.set_title('Sector Analysis (using merged data)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.axhline(0, color='black', linewidth=0.5)
ax4.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax4.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

# Add merge code annotation
ax3.text(0.02, 0.02, "result = pd.merge(df_prices, df_info, on='Ticker', how='inner')",
         transform=ax3.transAxes, fontsize=8, family='monospace', va='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
