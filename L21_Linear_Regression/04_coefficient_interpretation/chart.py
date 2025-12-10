"""Coefficient Interpretation - Understanding slope and intercept"""
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

x = np.linspace(-2, 20, 100)
slope = 0.5
intercept = 3
y = intercept + slope * x

ax.plot(x, y, color=MLBLUE, linewidth=2.5, label=f'y = {intercept} + {slope}x')

# Show slope visualization (rise over run)
x1, x2 = 8, 14
y1, y2 = intercept + slope * x1, intercept + slope * x2

ax.plot([x1, x2], [y1, y1], color=MLRED, linewidth=2.5, linestyle='--')
ax.plot([x2, x2], [y1, y2], color=MLGREEN, linewidth=2.5, linestyle='--')

# Add annotations with arrows
ax.annotate('', xy=(x2, y1), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='<->', color=MLRED, lw=2))
ax.text((x1 + x2)/2, y1 - 0.8, r'$\Delta x = 6$', ha='center', fontsize=11, color=MLRED, fontweight='bold')

ax.annotate('', xy=(x2, y2), xytext=(x2, y1),
            arrowprops=dict(arrowstyle='<->', color=MLGREEN, lw=2))
ax.text(x2 + 0.8, (y1 + y2)/2, r'$\Delta y = 3$', va='center', fontsize=11, color=MLGREEN, fontweight='bold')

# Mark intercept
ax.axhline(0, color='gray', linewidth=1)
ax.axvline(0, color='gray', linewidth=1)
ax.scatter([0], [intercept], c=MLORANGE, s=150, zorder=5, edgecolors='black')
ax.annotate(f'Intercept = {intercept}\n(y when x = 0)', xy=(0, intercept), xytext=(3, 4.5),
            fontsize=10, arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))

# Add slope formula
ax.text(15, 5, r'Slope = $\frac{\Delta y}{\Delta x} = \frac{3}{6} = 0.5$',
        fontsize=12, bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

ax.set_title('Interpreting Coefficients: Slope and Intercept', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Market Return (%)', fontsize=10)
ax.set_ylabel('Stock Return (%)', fontsize=10)
ax.set_xlim(-3, 22)
ax.set_ylim(-1, 14)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
