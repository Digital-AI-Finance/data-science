"""Decomposition Summary - Additive model components"""
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
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)
n = 100

trend = np.linspace(100, 140, n)
seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 30)
weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
noise = np.random.randn(n) * 3

fig, ax = plt.subplots(figsize=(10, 6))

components = {'Trend': trend, 'Seasonal': seasonal, 'Weekly': weekly, 'Noise': noise}
y_offset = 0
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]

for (name, data), color in zip(components.items(), colors):
    normalized = (data - data.mean()) / (data.max() - data.min() + 0.01) * 20
    ax.plot(range(n), normalized + y_offset, color=color, linewidth=2, label=name)
    ax.axhline(y_offset, color='gray', linewidth=0.5, linestyle=':')
    y_offset += 25

ax.set_xlabel('Days', fontsize=10)
ax.set_ylabel('Component (normalized)', fontsize=10)
ax.set_title('Decomposition: Additive Model', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.legend(fontsize=10, loc='upper right')
ax.set_yticks([])
ax.grid(alpha=0.3)

ax.text(50, 85, 'Y = Trend + Seasonal + Cyclical + Noise',
        ha='center', fontsize=11, fontweight='bold', color=MLPURPLE,
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.3))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
