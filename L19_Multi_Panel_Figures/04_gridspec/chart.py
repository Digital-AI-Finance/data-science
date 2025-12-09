"""GridSpec - Advanced layout control"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

fig = plt.figure(figsize=(14, 10))
fig.suptitle('GridSpec: Fine-Grained Layout Control', fontsize=14, fontweight='bold', color=MLPURPLE)

# Create GridSpec with custom ratios
gs = gridspec.GridSpec(3, 3, figure=fig,
                       width_ratios=[2, 1, 1],
                       height_ratios=[2, 1, 1],
                       hspace=0.3, wspace=0.3)

# Generate financial data
days = 252
prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.015))
returns = np.diff(np.log(prices)) * 100

# Main price chart (large, top-left)
ax_main = fig.add_subplot(gs[0, 0])
ax_main.plot(prices, color=MLBLUE, linewidth=2)
ax_main.fill_between(range(len(prices)), prices.min(), prices, alpha=0.2, color=MLBLUE)
ax_main.set_title('GridSpec: gs[0, 0]', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_main.set_ylabel('Price ($)', fontsize=10)
ax_main.grid(alpha=0.3)

# Return histogram (right of main)
ax_hist = fig.add_subplot(gs[0, 1:])
ax_hist.hist(returns, bins=30, color=MLGREEN, alpha=0.7, edgecolor='black')
ax_hist.axvline(np.mean(returns), color=MLRED, linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
ax_hist.set_title('gs[0, 1:] - Spans 2 columns', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_hist.set_xlabel('Return (%)', fontsize=10)
ax_hist.legend(fontsize=8)
ax_hist.grid(alpha=0.3)

# Rolling volatility (middle left)
ax_vol = fig.add_subplot(gs[1, 0])
rolling_vol = pd.Series(returns).rolling(20).std() * np.sqrt(252)
ax_vol.plot(rolling_vol, color=MLORANGE, linewidth=1.5)
ax_vol.set_title('gs[1, 0]', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_vol.set_ylabel('Volatility (%)', fontsize=10)
ax_vol.grid(alpha=0.3)

# Drawdown chart (middle center)
ax_dd = fig.add_subplot(gs[1, 1])
cummax = np.maximum.accumulate(prices)
drawdown = (prices - cummax) / cummax * 100
ax_dd.fill_between(range(len(drawdown)), 0, drawdown, color=MLRED, alpha=0.5)
ax_dd.set_title('gs[1, 1]', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_dd.set_ylabel('DD (%)', fontsize=9)
ax_dd.grid(alpha=0.3)

# Summary stats (middle right)
ax_stats = fig.add_subplot(gs[1, 2])
ax_stats.axis('off')
stats_text = f'''Statistics:
Mean Return: {np.mean(returns):.3f}%
Std Dev: {np.std(returns):.3f}%
Max DD: {drawdown.min():.1f}%
Sharpe: {np.mean(returns)/np.std(returns)*np.sqrt(252):.2f}'''
ax_stats.text(0.1, 0.8, stats_text, transform=ax_stats.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax_stats.set_title('gs[1, 2]', fontsize=10, fontweight='bold', color=MLPURPLE)

# Bottom spanning plot
ax_bottom = fig.add_subplot(gs[2, :])
# Monthly performance
months = pd.date_range('2024-01-01', periods=12, freq='ME')
monthly_ret = np.random.normal(0.8, 2.5, 12)
colors = [MLGREEN if r > 0 else MLRED for r in monthly_ret]
ax_bottom.bar(range(12), monthly_ret, color=colors, edgecolor='black', linewidth=0.5)
ax_bottom.set_xticks(range(12))
ax_bottom.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax_bottom.axhline(0, color='gray', linewidth=1)
ax_bottom.set_title('gs[2, :] - Full width bottom row', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_bottom.set_ylabel('Return (%)', fontsize=10)
ax_bottom.grid(alpha=0.3, axis='y')

plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
