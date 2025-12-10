"""Annotations and Text - Highlighting key points"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

prices = 100 + np.cumsum(np.random.randn(100) * 2)
ax.plot(prices, color=MLBLUE, linewidth=2)

# Find max and min
max_idx = np.argmax(prices)
min_idx = np.argmin(prices)

ax.scatter([max_idx], [prices[max_idx]], color=MLGREEN, s=100, zorder=5)
ax.scatter([min_idx], [prices[min_idx]], color=MLRED, s=100, zorder=5)

ax.annotate(f'Max: ${prices[max_idx]:.0f}', xy=(max_idx, prices[max_idx]),
            xytext=(max_idx - 20, prices[max_idx] + 8),
            fontsize=10, fontweight='bold', color=MLGREEN,
            arrowprops=dict(arrowstyle='->', color=MLGREEN))

ax.annotate(f'Min: ${prices[min_idx]:.0f}', xy=(min_idx, prices[min_idx]),
            xytext=(min_idx + 10, prices[min_idx] - 8),
            fontsize=10, fontweight='bold', color=MLRED,
            arrowprops=dict(arrowstyle='->', color=MLRED))

ax.text(0.5, 0.95, 'Annotations highlight key points',
        transform=ax.transAxes, fontsize=10, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

ax.set_title('Annotations and Text', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
