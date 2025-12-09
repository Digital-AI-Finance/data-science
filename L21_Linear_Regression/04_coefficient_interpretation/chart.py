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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Interpreting Regression Coefficients', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Slope interpretation
ax1 = axes[0, 0]

x = np.linspace(0, 20, 100)
y = 2 + 0.5 * x

ax1.plot(x, y, color=MLBLUE, linewidth=2.5)

# Show slope visualization
x1, x2 = 8, 12
y1, y2 = 2 + 0.5 * 8, 2 + 0.5 * 12

ax1.plot([x1, x2], [y1, y1], color=MLRED, linewidth=2, linestyle='--')
ax1.plot([x2, x2], [y1, y2], color=MLGREEN, linewidth=2, linestyle='--')

# Add annotations
ax1.annotate('', xy=(x2, y1), xytext=(x1, y1),
             arrowprops=dict(arrowstyle='<->', color=MLRED, lw=2))
ax1.text((x1 + x2)/2, y1 - 0.5, r'$\Delta x = 4$', ha='center', fontsize=11, color=MLRED)

ax1.annotate('', xy=(x2, y2), xytext=(x2, y1),
             arrowprops=dict(arrowstyle='<->', color=MLGREEN, lw=2))
ax1.text(x2 + 0.5, (y1 + y2)/2, r'$\Delta y = 2$', va='center', fontsize=11, color=MLGREEN)

ax1.text(15, 3, r'Slope = $\frac{\Delta y}{\Delta x} = \frac{2}{4} = 0.5$',
         fontsize=12, bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

ax1.set_title('Slope: Change in Y per Unit Change in X', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('X (Risk %)', fontsize=10)
ax1.set_ylabel('Y (Return %)', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Intercept interpretation
ax2 = axes[0, 1]

x = np.linspace(-2, 15, 100)
y = 3 + 0.4 * x

ax2.plot(x, y, color=MLBLUE, linewidth=2.5)
ax2.axhline(0, color='gray', linewidth=1)
ax2.axvline(0, color='gray', linewidth=1)

# Mark intercept
ax2.scatter([0], [3], c=MLORANGE, s=150, zorder=5, edgecolors='black')
ax2.annotate('Intercept = 3.0\n(y when x = 0)', xy=(0, 3), xytext=(3, 4.5),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))

ax2.set_title('Intercept: Y Value When X = 0', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('X (Market Risk Premium %)', fontsize=10)
ax2.set_ylabel('Y (Expected Return %)', fontsize=10)
ax2.set_xlim(-2, 15)
ax2.grid(alpha=0.3)

ax2.text(8, 2, 'In CAPM:\nIntercept = Risk-free rate',
         fontsize=10, bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 3: Different slopes comparison
ax3 = axes[1, 0]

x = np.linspace(0, 10, 100)
slopes = [0.2, 0.5, 1.0, 1.5]
colors = [MLLAVENDER, MLBLUE, MLGREEN, MLORANGE]

for slope, color in zip(slopes, colors):
    ax3.plot(x, 2 + slope * x, color=color, linewidth=2.5, label=f'Slope = {slope}')

ax3.set_title('Slope Comparison: Steeper = Stronger Effect', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Y', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

ax3.text(0.5, 0.95, 'Higher slope = X has more impact on Y',
         transform=ax3.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 4: Finance interpretation example
ax4 = axes[1, 1]
ax4.axis('off')

interpretation = '''
FINANCE EXAMPLE: CAPM Regression

Model: Stock Return = alpha + beta * Market Return

y = 0.5 + 1.2 * x

Coefficient Interpretations:

INTERCEPT (alpha = 0.5%):
- Expected return when market return = 0
- Positive alpha: stock outperforms (risk-adjusted)
- Negative alpha: stock underperforms

SLOPE (beta = 1.2):
- For every 1% market move, stock moves 1.2%
- beta > 1: more volatile than market
- beta < 1: less volatile than market
- beta = 1: moves with market

Example Predictions:
- Market +5% -> Stock = 0.5 + 1.2(5) = +6.5%
- Market -3% -> Stock = 0.5 + 1.2(-3) = -3.1%
'''

ax4.text(0.02, 0.95, interpretation, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Finance Application: CAPM Beta', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
