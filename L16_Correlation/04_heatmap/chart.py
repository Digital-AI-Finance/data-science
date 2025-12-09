"""Correlation Heatmap - Visualizing correlation matrices"""
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

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Correlation Heatmaps', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate correlated asset returns
n = 252
assets = ['SPY', 'QQQ', 'GLD', 'TLT', 'VNQ', 'EEM']

# Create correlation structure
base_market = np.random.normal(0, 1, n)
returns = pd.DataFrame({
    'SPY': base_market + np.random.normal(0, 0.3, n),
    'QQQ': 1.2 * base_market + np.random.normal(0, 0.4, n),  # High correlation with SPY
    'GLD': 0.1 * base_market + np.random.normal(0, 0.8, n),  # Low correlation
    'TLT': -0.3 * base_market + np.random.normal(0, 0.5, n),  # Negative correlation
    'VNQ': 0.6 * base_market + np.random.normal(0, 0.5, n),
    'EEM': 0.7 * base_market + np.random.normal(0, 0.6, n),
})

corr = returns.corr()

# Plot 1: Basic heatmap
ax1 = axes[0]
im = ax1.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label('Correlation', fontsize=10)

# Set ticks
ax1.set_xticks(range(len(assets)))
ax1.set_yticks(range(len(assets)))
ax1.set_xticklabels(assets, fontsize=10)
ax1.set_yticklabels(assets, fontsize=10)

# Add values
for i in range(len(assets)):
    for j in range(len(assets)):
        text = ax1.text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha='center', va='center', fontsize=9,
                        color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')

ax1.set_title('Asset Correlation Matrix', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Lower triangle only (no redundancy)
ax2 = axes[1]

# Mask upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
masked_corr = corr.copy()
masked_corr = masked_corr.where(~mask)

im2 = ax2.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Mask upper triangle visually
for i in range(len(assets)):
    for j in range(len(assets)):
        if j > i:
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='white'))
        else:
            text = ax2.text(j, i, f'{corr.iloc[i, j]:.2f}',
                            ha='center', va='center', fontsize=9,
                            color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')

cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label('Correlation', fontsize=10)

ax2.set_xticks(range(len(assets)))
ax2.set_yticks(range(len(assets)))
ax2.set_xticklabels(assets, fontsize=10)
ax2.set_yticklabels(assets, fontsize=10)
ax2.set_title('Lower Triangle (Cleaner)', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
