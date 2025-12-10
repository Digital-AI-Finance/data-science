"""Line Styles and Markers - Different visual options"""
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
MLORANGE = '#FF7F0E'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 50)
y = np.sin(x) + np.random.normal(0, 0.1, 50)

ax.plot(x[:20], y[:20], 'o-', color=MLBLUE, linewidth=2, markersize=8, label='Solid + circle')
ax.plot(x[:20], y[:20]+1, 's--', color=MLGREEN, linewidth=2, markersize=6, label='Dashed + square')
ax.plot(x[:20], y[:20]+2, '^:', color=MLRED, linewidth=2, markersize=6, label='Dotted + triangle')
ax.plot(x[:20], y[:20]+3, 'D-.', color=MLORANGE, linewidth=2, markersize=6, label='Dash-dot + diamond')

ax.set_title('Line Styles and Markers', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
