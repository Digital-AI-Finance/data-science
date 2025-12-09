"""Annotations - Adding context to charts"""
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
fig.suptitle('Annotations and Labels', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Arrow annotations
ax1 = axes[0, 0]
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax1.plot(x, y, color=MLBLUE, linewidth=2)

# Different arrow styles
ax1.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.3),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLGREEN, lw=2))

ax1.annotate('Trough', xy=(3*np.pi/2, -1), xytext=(3*np.pi/2 + 1, -0.5),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='fancy', color=MLRED, connectionstyle='arc3,rad=0.3'))

ax1.annotate('Zero crossing', xy=(np.pi, 0), xytext=(np.pi - 1.5, 0.5),
             fontsize=9,
             arrowprops=dict(arrowstyle='wedge', color=MLORANGE))

ax1.set_title('Arrow Styles', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.grid(alpha=0.3)

# Plot 2: Text boxes
ax2 = axes[0, 1]
prices = 100 + np.cumsum(np.random.randn(100) * 2)
ax2.plot(prices, color=MLBLUE, linewidth=2)

# Different bbox styles
ax2.text(10, prices.max() - 5, 'Bull Market', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=MLGREEN, alpha=0.7))

ax2.text(60, prices.min() + 5, 'Correction', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='square', facecolor=MLRED, alpha=0.7, edgecolor='black'))

ax2.text(80, np.mean(prices), 'Recovery', fontsize=10,
         bbox=dict(boxstyle='rarrow', facecolor=MLORANGE, alpha=0.7))

ax2.set_title('Text Boxes (bbox)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3)

# Plot 3: Event markers
ax3 = axes[1, 0]
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 100 + np.cumsum(np.random.randn(100) * 2)
ax3.plot(dates, prices, color=MLBLUE, linewidth=2)

# Mark events
events = [(20, 'Earnings', MLGREEN), (45, 'Fed Meeting', MLRED), (75, 'Dividend', MLORANGE)]
for idx, label, color in events:
    ax3.axvline(dates[idx], color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.scatter([dates[idx]], [prices[idx]], color=color, s=100, zorder=5)
    ax3.annotate(label, xy=(dates[idx], prices[idx]),
                 xytext=(5, 10), textcoords='offset points',
                 fontsize=9, color=color, fontweight='bold')

ax3.set_title('Event Markers', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(alpha=0.3)

# Plot 4: Statistics annotations
ax4 = axes[1, 1]
returns = np.random.normal(0.05, 2, 252)
ax4.hist(returns, bins=30, color=MLBLUE, alpha=0.7, edgecolor='black')

mean_ret = np.mean(returns)
std_ret = np.std(returns)
var_95 = np.percentile(returns, 5)

ax4.axvline(mean_ret, color=MLGREEN, linewidth=2.5, label=f'Mean: {mean_ret:.2f}%')
ax4.axvline(var_95, color=MLRED, linewidth=2.5, linestyle='--', label=f'VaR 95%: {var_95:.2f}%')

# Add statistics box
stats_text = f'Mean: {mean_ret:.2f}%\nStd: {std_ret:.2f}%\nVaR(95%): {var_95:.2f}%'
ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLPURPLE, alpha=0.9))

ax4.set_title('Statistics Summary Box', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Return (%)', fontsize=10)
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
