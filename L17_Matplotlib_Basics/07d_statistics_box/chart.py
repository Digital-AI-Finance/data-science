"""Statistics Summary Box - Key metrics annotation"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLBLUE = '#0066CC'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

returns = np.random.normal(0.05, 2, 252)
ax.hist(returns, bins=30, color=MLBLUE, alpha=0.7, edgecolor='black')

mean_ret = np.mean(returns)
std_ret = np.std(returns)
var_95 = np.percentile(returns, 5)

ax.axvline(mean_ret, color=MLGREEN, linewidth=2.5, label=f'Mean: {mean_ret:.2f}%')
ax.axvline(var_95, color=MLRED, linewidth=2.5, linestyle='--', label=f'VaR 95%: {var_95:.2f}%')

# Add statistics box
stats_text = f'Mean: {mean_ret:.2f}%\nStd: {std_ret:.2f}%\nVaR(95%): {var_95:.2f}%'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLPURPLE, alpha=0.9))

ax.set_title('Statistics Summary Box', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Return (%)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
