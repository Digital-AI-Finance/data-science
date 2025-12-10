"""Line with Markers - Adding data point markers"""
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLORANGE = '#FF7F0E'

fig, ax = plt.subplots(figsize=(10, 6))

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
values = [100, 105, 102, 110, 108, 115]

ax.plot(months, values, 'o-', color=MLORANGE, linewidth=2, markersize=10,
        markerfacecolor='white', markeredgewidth=2)

for i, v in enumerate(values):
    ax.annotate(f'{v}', (i, v), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

ax.set_title('Line with Markers and Labels', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Month', fontsize=10)
ax.set_ylabel('Value', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
