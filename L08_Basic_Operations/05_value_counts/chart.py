"""Value Counts - Frequency analysis with value_counts()"""
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

# Create sample stock data
np.random.seed(42)
n = 200

# Sectors
sectors = np.random.choice(['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer'],
                           size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])

# Market cap categories
market_caps = np.random.choice(['Large Cap', 'Mid Cap', 'Small Cap'],
                               size=n, p=[0.35, 0.40, 0.25])

# Trading signals
signals = np.random.choice(['Buy', 'Hold', 'Sell'],
                           size=n, p=[0.25, 0.50, 0.25])

df = pd.DataFrame({'Sector': sectors, 'MarketCap': market_caps, 'Signal': signals})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Frequency Analysis with value_counts()', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Sector distribution
ax1 = axes[0, 0]
sector_counts = df['Sector'].value_counts()
colors_sector = [MLBLUE, MLORANGE, MLGREEN, MLRED, MLPURPLE]
bars1 = ax1.bar(sector_counts.index, sector_counts.values, color=colors_sector, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Count', fontsize=10)
ax1.set_title("df['Sector'].value_counts()", fontsize=11, color=MLBLUE, family='monospace')
ax1.tick_params(axis='x', rotation=30)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

# Plot 2: Normalized (percentages)
ax2 = axes[0, 1]
sector_pct = df['Sector'].value_counts(normalize=True) * 100
bars2 = ax2.bar(sector_pct.index, sector_pct.values, color=colors_sector, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Percentage (%)', fontsize=10)
ax2.set_title("df['Sector'].value_counts(normalize=True)", fontsize=11, color=MLORANGE, family='monospace')
ax2.tick_params(axis='x', rotation=30)
ax2.grid(axis='y', alpha=0.3)

for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

# Plot 3: Pie chart for market cap
ax3 = axes[1, 0]
mc_counts = df['MarketCap'].value_counts()
colors_mc = [MLGREEN, MLBLUE, MLORANGE]
wedges, texts, autotexts = ax3.pie(mc_counts.values, labels=mc_counts.index, autopct='%1.1f%%',
                                   colors=colors_mc, explode=[0.02, 0.02, 0.02],
                                   textprops={'fontsize': 10})
ax3.set_title("df['MarketCap'].value_counts() as Pie", fontsize=11, color=MLGREEN)

# Plot 4: Grouped analysis
ax4 = axes[1, 1]
# Cross-tabulation
signal_by_sector = pd.crosstab(df['Sector'], df['Signal'])
signal_by_sector_pct = signal_by_sector.div(signal_by_sector.sum(axis=1), axis=0) * 100

x = np.arange(len(signal_by_sector.index))
width = 0.25

bars_buy = ax4.bar(x - width, signal_by_sector_pct['Buy'], width, label='Buy', color=MLGREEN, alpha=0.7)
bars_hold = ax4.bar(x, signal_by_sector_pct['Hold'], width, label='Hold', color=MLBLUE, alpha=0.7)
bars_sell = ax4.bar(x + width, signal_by_sector_pct['Sell'], width, label='Sell', color=MLRED, alpha=0.7)

ax4.set_xlabel('Sector', fontsize=10)
ax4.set_ylabel('Percentage (%)', fontsize=10)
ax4.set_title('Signal Distribution by Sector', fontsize=11, color=MLPURPLE, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(signal_by_sector.index, rotation=30)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
