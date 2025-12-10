"""Scatter with Regression - Risk vs Return"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

risk = np.random.uniform(5, 25, 30)
returns = 2 + 0.4 * risk + np.random.normal(0, 2, 30)

ax.scatter(risk, returns, c=MLGREEN, s=80, alpha=0.7, edgecolors='black')

# Regression line
z = np.polyfit(risk, returns, 1)
ax.plot(np.sort(risk), np.poly1d(z)(np.sort(risk)), color=MLRED, linewidth=2.5,
        label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

ax.set_title('Risk vs Return (with regression)', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Risk (%)', fontsize=10)
ax.set_ylabel('Expected Return (%)', fontsize=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
