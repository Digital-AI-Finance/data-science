"""Fill Between - Shading area between lines"""
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

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.sin(x) + 0.5

ax.plot(x, y1, color=MLBLUE, linewidth=2, label='Lower bound')
ax.plot(x, y2, color=MLGREEN, linewidth=2, label='Upper bound')
ax.fill_between(x, y1, y2, alpha=0.3, color=MLPURPLE, label='Confidence band')

ax.set_title('Fill Between Lines', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
