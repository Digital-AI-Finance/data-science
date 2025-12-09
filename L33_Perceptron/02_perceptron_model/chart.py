"""Perceptron Model - Mathematical Definition"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('The Perceptron: Mathematical Model', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Perceptron formula
ax1 = axes[0, 0]
ax1.axis('off')

formula = '''
PERCEPTRON MATHEMATICAL MODEL

INPUTS:
-------
x = [x1, x2, ..., xn]   Features


PARAMETERS:
-----------
w = [w1, w2, ..., wn]   Weights
b                        Bias


COMPUTATION:
------------
Step 1: Weighted sum
        z = w1*x1 + w2*x2 + ... + wn*xn + b
        z = w . x + b   (dot product)

Step 2: Activation (threshold)
        y = 1   if z >= 0
        y = 0   if z < 0


COMPACT FORM:
-------------
y = step(w . x + b)

where step(z) = 1 if z >= 0, else 0


EXAMPLE:
--------
x = [2, 3]
w = [0.5, -0.3]
b = -0.5

z = 0.5*2 + (-0.3)*3 + (-0.5)
z = 1 - 0.9 - 0.5 = -0.4

y = step(-0.4) = 0
'''

ax1.text(0.02, 0.98, formula, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Perceptron Formula', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Perceptron diagram
ax2 = axes[0, 1]
ax2.axis('off')

# Draw perceptron
# Inputs
inputs = ['x1', 'x2', 'x3', '1']
y_positions = [0.8, 0.6, 0.4, 0.2]
weights = ['w1', 'w2', 'w3', 'b']

for i, (inp, y, w) in enumerate(zip(inputs, y_positions, weights)):
    # Input node
    circle = plt.Circle((0.15, y), 0.05, color=MLBLUE, ec='black')
    ax2.add_patch(circle)
    ax2.text(0.15, y, inp, fontsize=10, fontweight='bold', ha='center', va='center', color='white')

    # Connection line with weight
    ax2.plot([0.2, 0.45], [y, 0.5], color='black', lw=2)
    mid_x = (0.2 + 0.45) / 2
    mid_y = (y + 0.5) / 2
    ax2.text(mid_x-0.03, mid_y+0.02, w, fontsize=9, color=MLORANGE, fontweight='bold')

# Sum node
circle_sum = plt.Circle((0.5, 0.5), 0.08, color=MLGREEN, ec='black', linewidth=2)
ax2.add_patch(circle_sum)
ax2.text(0.5, 0.5, 'Sum', fontsize=9, fontweight='bold', ha='center', va='center')

# Arrow to activation
ax2.annotate('', xy=(0.65, 0.5), xytext=(0.58, 0.5),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Activation box
bbox = dict(boxstyle='round', facecolor=MLORANGE, alpha=0.8)
ax2.text(0.72, 0.5, 'Step\nFunction', fontsize=9, fontweight='bold', ha='center', va='center', bbox=bbox)

# Output arrow
ax2.annotate('', xy=(0.9, 0.5), xytext=(0.8, 0.5),
             arrowprops=dict(arrowstyle='->', color=MLRED, lw=3))

# Output node
circle_out = plt.Circle((0.95, 0.5), 0.05, color=MLRED, ec='black')
ax2.add_patch(circle_out)
ax2.text(0.95, 0.5, 'y', fontsize=11, fontweight='bold', ha='center', va='center', color='white')

# Labels
ax2.text(0.07, 0.9, 'Inputs', fontsize=10, color=MLBLUE, fontweight='bold')
ax2.text(0.5, 0.3, 'Weighted\nSum', fontsize=9, ha='center', color=MLGREEN)
ax2.text(0.95, 0.35, 'Output', fontsize=9, ha='center', color=MLRED)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Perceptron Architecture', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Step function
ax3 = axes[1, 0]

z = np.linspace(-3, 3, 100)
step = np.where(z >= 0, 1, 0)

ax3.plot(z, step, color=MLBLUE, linewidth=3)
ax3.axhline(0, color='black', linewidth=0.5)
ax3.axvline(0, color='black', linewidth=0.5)

ax3.scatter([0], [0], c=MLBLUE, s=100, zorder=5)
ax3.scatter([0], [1], c=MLBLUE, s=100, zorder=5, facecolors='white', edgecolors=MLBLUE, linewidths=2)

ax3.set_title('Step Function (Heaviside)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('z = w.x + b')
ax3.set_ylabel('Output y')
ax3.grid(alpha=0.3)
ax3.set_ylim(-0.2, 1.2)

# Annotate
ax3.text(1.5, 0.8, 'y = 1\n(fire)', fontsize=10, color=MLGREEN)
ax3.text(-2.5, 0.2, 'y = 0\n(no fire)', fontsize=10, color=MLRED)

# Plot 4: Bias role
ax4 = axes[1, 1]
ax4.axis('off')

bias = '''
THE ROLE OF BIAS

WHAT IS BIAS?
-------------
The bias b shifts the decision boundary.
It's like a "threshold" for activation.


WITHOUT BIAS:
-------------
y = step(w . x)
Decision boundary passes through origin.


WITH BIAS:
----------
y = step(w . x + b)
Decision boundary can be anywhere!


GEOMETRIC INTERPRETATION:
-------------------------
y = step(w1*x1 + w2*x2 + b)

Decision boundary: w1*x1 + w2*x2 + b = 0

This is a LINE (or hyperplane in higher D)

- w = [w1, w2] determines the direction
- b determines the offset from origin


EXAMPLE:
--------
w = [1, 1], b = -1.5

Boundary: x1 + x2 - 1.5 = 0
          x1 + x2 = 1.5

Points with x1 + x2 > 1.5 -> class 1
Points with x1 + x2 < 1.5 -> class 0
'''

ax4.text(0.02, 0.98, bias, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Role of Bias', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
