"""Horizontal Bar - Sector Returns"""
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

fig, ax = plt.subplots(figsize=(10, 6))

sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy']
returns = [12.5, 8.3, 6.7, 5.2, -2.1]
colors = [MLGREEN if r > 0 else MLRED for r in returns]

bars = ax.barh(sectors, returns, color=colors, alpha=0.8, edgecolor='black')
ax.axvline(0, color='black', linewidth=1)

for bar, val in zip(bars, returns):
    x_pos = val + 0.5 if val > 0 else val - 1
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
            va='center', fontsize=9, fontweight='bold')

ax.set_title('Horizontal Bar: Sector Returns', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Return (%)', fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
