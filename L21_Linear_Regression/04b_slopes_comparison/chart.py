"""Slopes Comparison - Different beta values"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(-5, 10, 100)
intercept = 2

# Different beta values
betas = [
    (0.5, '#2CA02C', 'Beta = 0.5 (Defensive)'),
    (1.0, '#0066CC', 'Beta = 1.0 (Market)'),
    (1.5, '#FF7F0E', 'Beta = 1.5 (Aggressive)'),
    (2.0, '#D62728', 'Beta = 2.0 (Very Aggressive)')
]

for beta, color, label in betas:
    ax.plot(x, intercept + beta * x, color=color, linewidth=2.5, label=label)

ax.axhline(0, color='gray', linewidth=1, linestyle='--')
ax.axvline(0, color='gray', linewidth=1, linestyle='--')

ax.set_title('Higher Slope = Stronger Market Sensitivity', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Market Return (%)', fontsize=10)
ax.set_ylabel('Stock Return (%)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(-5, 10)
ax.set_ylim(-8, 25)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
