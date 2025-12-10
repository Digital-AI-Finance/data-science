"""Transparency (alpha) - Layering with opacity"""
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

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 50)
y = np.sin(x) + np.random.normal(0, 0.1, 50)

for alpha, offset in [(1.0, 0), (0.7, 0.5), (0.4, 1.0), (0.2, 1.5)]:
    ax.fill_between(x, y + offset, y + offset + 0.4, alpha=alpha, color=MLBLUE,
                    label=f'alpha = {alpha}')

ax.set_title('Transparency (alpha)', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
