"""Sector Analysis - Real-world groupby application"""
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

# Create comprehensive stock dataset
np.random.seed(42)
sectors = {
    'Technology': {'count': 15, 'return': 0.12, 'vol': 0.25, 'pe': 35},
    'Finance': {'count': 12, 'return': 0.06, 'vol': 0.18, 'pe': 15},
    'Healthcare': {'count': 10, 'return': 0.08, 'vol': 0.20, 'pe': 25},
    'Energy': {'count': 8, 'return': 0.03, 'vol': 0.30, 'pe': 12},
    'Consumer': {'count': 10, 'return': 0.07, 'vol': 0.15, 'pe': 20},
}

data = []
for sector, params in sectors.items():
    for i in range(params['count']):
        data.append({
            'Sector': sector,
            'Return': np.random.normal(params['return'], params['vol']/3),
            'Volatility': np.random.normal(params['vol'], 0.05),
            'PE_Ratio': np.random.normal(params['pe'], 5),
            'MarketCap': np.random.lognormal(mean=np.log(50), sigma=1)
        })

df = pd.DataFrame(data)
df['MarketCap'] = df['MarketCap'].clip(10, 500)  # Billion USD

# GroupBy analysis
sector_stats = df.groupby('Sector').agg({
    'Return': ['mean', 'std'],
    'Volatility': 'mean',
    'PE_Ratio': 'mean',
    'MarketCap': ['mean', 'sum']
}).round(3)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sector Analysis Using GroupBy', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Risk-Return by sector
ax1 = axes[0, 0]
sector_names = list(sectors.keys())
colors = [MLBLUE, MLORANGE, MLGREEN, MLRED, MLPURPLE]

for sector, color in zip(sector_names, colors):
    sector_data = df[df['Sector'] == sector]
    ax1.scatter(sector_data['Volatility'] * 100, sector_data['Return'] * 100,
                alpha=0.6, s=50, color=color, label=sector)

# Add sector means
for sector, color in zip(sector_names, colors):
    mean_vol = df[df['Sector'] == sector]['Volatility'].mean() * 100
    mean_ret = df[df['Sector'] == sector]['Return'].mean() * 100
    ax1.scatter(mean_vol, mean_ret, color=color, s=200, marker='*', edgecolor='black', linewidth=2)

ax1.set_xlabel('Volatility (%)', fontsize=10)
ax1.set_ylabel('Return (%)', fontsize=10)
ax1.set_title('Risk-Return Profile by Sector', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.5)

# Plot 2: Mean metrics comparison
ax2 = axes[0, 1]
metrics = ['Return (%)', 'Volatility (%)', 'PE Ratio']
x = np.arange(len(sector_names))
width = 0.25

metric_data = {
    'Return (%)': df.groupby('Sector')['Return'].mean() * 100,
    'Volatility (%)': df.groupby('Sector')['Volatility'].mean() * 100,
    'PE Ratio': df.groupby('Sector')['PE_Ratio'].mean() / 3  # Scaled for display
}

for i, (metric, values) in enumerate(metric_data.items()):
    ax2.bar(x + i*width, values[sector_names], width, label=metric, alpha=0.7)

ax2.set_ylabel('Value (scaled)', fontsize=10)
ax2.set_title('Sector Metrics Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xticks(x + width)
ax2.set_xticklabels(sector_names, rotation=30, ha='right')
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Market Cap distribution by sector
ax3 = axes[1, 0]
sector_caps = df.groupby('Sector')['MarketCap'].sum()
colors_pie = [MLBLUE, MLORANGE, MLGREEN, MLRED, MLPURPLE]
wedges, texts, autotexts = ax3.pie(sector_caps, labels=sector_caps.index,
                                   autopct='%1.1f%%', colors=colors_pie,
                                   explode=[0.02]*5, textprops={'fontsize': 9})
ax3.set_title('Total Market Cap by Sector', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Box plot of returns by sector
ax4 = axes[1, 1]
sector_returns = [df[df['Sector'] == s]['Return'] * 100 for s in sector_names]
bp = ax4.boxplot(sector_returns, labels=sector_names, patch_artist=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax4.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax4.set_ylabel('Return (%)', fontsize=10)
ax4.set_title('Return Distribution by Sector', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.tick_params(axis='x', rotation=30)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
