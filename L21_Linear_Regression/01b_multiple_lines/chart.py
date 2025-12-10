"""Multiple Possible Lines - Which line fits best?"""
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

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

# Generate data
n = 30
x = np.random.uniform(5, 25, n)
y_true = 2 + 0.4 * x
y = y_true + np.random.normal(0, 1.5, n)

ax.scatter(x, y, c=MLBLUE, s=80, alpha=0.7, edgecolors='black', label='Data')

# Draw several possible lines
x_line = np.linspace(5, 25, 100)
lines = [
    (0.3, 4, MLRED, 'Line A: too flat'),
    (0.5, 0, MLORANGE, 'Line B: wrong intercept'),
    (0.4, 2, MLGREEN, 'Line C: best fit!')
]

for slope, intercept, color, label in lines:
    linestyle = '-' if color == MLGREEN else '--'
    linewidth = 2.5 if color == MLGREEN else 1.5
    ax.plot(x_line, intercept + slope * x_line, color=color, linewidth=linewidth,
            linestyle=linestyle, label=label)

ax.set_title('Which Line Fits Best?', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Risk (%)', fontsize=10)
ax.set_ylabel('Return (%)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
