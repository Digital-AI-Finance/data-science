"""Nested Plots - Inset axes and zoom windows"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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
fig.suptitle('Nested Plots: Insets and Zoom Windows', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
days = 252
prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.015))

# Plot 1: Zoom inset
ax1 = axes[0, 0]
ax1.plot(prices, color=MLBLUE, linewidth=2)
ax1.set_title('Zoom Inset (inset_axes)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Day', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.grid(alpha=0.3)

# Add inset for zoom
ax_inset = inset_axes(ax1, width="40%", height="35%", loc='upper left',
                       bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=ax1.transAxes)
zoom_start, zoom_end = 100, 150
ax_inset.plot(range(zoom_start, zoom_end), prices[zoom_start:zoom_end], color=MLBLUE, linewidth=2)
ax_inset.set_xlim(zoom_start, zoom_end)
ax_inset.set_title('Zoomed Region', fontsize=8, color=MLPURPLE)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(alpha=0.3)

# Mark zoom region
ax1.axvspan(zoom_start, zoom_end, alpha=0.2, color=MLORANGE)

# Plot 2: Pie inset in bar chart
ax2 = axes[0, 1]
sectors = ['Tech', 'Finance', 'Health', 'Energy', 'Consumer']
returns = [15.2, 8.5, 12.1, 6.3, 9.8]
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED, MLPURPLE]

ax2.bar(sectors, returns, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_title('Bar Chart with Pie Inset', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Return (%)', fontsize=10)
ax2.grid(alpha=0.3, axis='y')

# Add pie inset
ax_pie = inset_axes(ax2, width="35%", height="35%", loc='upper right')
weights = [0.35, 0.25, 0.20, 0.12, 0.08]
ax_pie.pie(weights, colors=colors, autopct='%1.0f%%', textprops={'fontsize': 7})
ax_pie.set_title('Portfolio\nWeights', fontsize=8, color=MLPURPLE)

# Plot 3: Secondary data inset
ax3 = axes[1, 0]
returns = np.diff(np.log(prices)) * 100

ax3.plot(returns, color=MLBLUE, linewidth=1.5, alpha=0.7)
ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_title('Returns with Distribution Inset', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Day', fontsize=10)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.grid(alpha=0.3)

# Add histogram inset
ax_hist = inset_axes(ax3, width="30%", height="40%", loc='upper right')
ax_hist.hist(returns, bins=25, color=MLGREEN, alpha=0.7, edgecolor='white', orientation='horizontal')
ax_hist.axhline(0, color=MLRED, linestyle='--', linewidth=1.5)
ax_hist.tick_params(labelsize=7)
ax_hist.set_xlabel('Count', fontsize=7)

# Plot 4: Multiple insets
ax4 = axes[1, 1]

# Multi-asset comparison
assets = ['AAPL', 'MSFT', 'GOOGL']
asset_prices = {}
for asset in assets:
    asset_prices[asset] = 100 + np.cumsum(np.random.randn(100) * 2)

for asset, color in zip(assets, [MLBLUE, MLGREEN, MLORANGE]):
    ax4.plot(asset_prices[asset] / asset_prices[asset][0] * 100, color=color, linewidth=2, label=asset)

ax4.axhline(100, color='gray', linestyle='--', linewidth=1)
ax4.set_title('Multi-Asset with Stat Insets', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Day', fontsize=10)
ax4.set_ylabel('Indexed Price', fontsize=10)
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(alpha=0.3)

# Add statistics inset
ax_stats = inset_axes(ax4, width="25%", height="30%", loc='lower right')
ax_stats.axis('off')
final_returns = [(asset_prices[a][-1] / asset_prices[a][0] - 1) * 100 for a in assets]
stats_text = '\n'.join([f'{a}: {r:+.1f}%' for a, r in zip(assets, final_returns)])
ax_stats.text(0.1, 0.9, 'Total Return:\n' + stats_text, transform=ax_stats.transAxes, fontsize=8,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
