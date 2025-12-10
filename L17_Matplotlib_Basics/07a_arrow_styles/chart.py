"""Arrow Styles - Different annotation arrows"""
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

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, color=MLBLUE, linewidth=2)

# Different arrow styles
ax.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.3),
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=MLGREEN, lw=2))

ax.annotate('Trough', xy=(3*np.pi/2, -1), xytext=(3*np.pi/2 + 1, -0.5),
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='fancy', color=MLRED, connectionstyle='arc3,rad=0.3'))

ax.annotate('Zero crossing', xy=(np.pi, 0), xytext=(np.pi - 1.5, 0.5),
            fontsize=9,
            arrowprops=dict(arrowstyle='wedge', color=MLORANGE))

ax.set_title('Arrow Styles', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
