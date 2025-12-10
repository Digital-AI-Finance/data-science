"""Price + Volume - Dual axis chart"""
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

n = 100
prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015))
volume = np.random.randint(100, 500, n).astype(float)
volume[prices > np.roll(prices, 1)] *= 1.2

ax.plot(prices, color=MLBLUE, linewidth=2)
ax_vol = ax.twinx()
colors = [MLGREEN if prices[i] > prices[i-1] else MLRED for i in range(1, n)]
colors = [MLGREEN] + colors
ax_vol.bar(range(n), volume, color=colors, alpha=0.3, width=1)
ax_vol.set_ylabel('Volume', fontsize=10, color='gray')
ax_vol.set_ylim(0, volume.max() * 3)

ax.set_title('Price + Volume', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Day', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
