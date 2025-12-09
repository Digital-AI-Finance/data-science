"""XOR Problem - The Famous Limitation"""
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
fig.suptitle('The XOR Problem: Perceptron\'s Famous Limitation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: AND gate (solvable)
ax1 = axes[0, 0]

# AND truth table
and_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_labels = np.array([0, 0, 0, 1])

colors = [MLBLUE if l == 0 else MLRED for l in and_labels]
ax1.scatter(and_points[:, 0], and_points[:, 1], c=colors, s=200, edgecolors='black', linewidths=2)

# Label points
for i, (x, y) in enumerate(and_points):
    ax1.annotate(f'{int(and_labels[i])}', (x, y), xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold')

# Decision boundary
x_line = np.linspace(-0.5, 1.5, 100)
y_line = 1.5 - x_line
ax1.plot(x_line, y_line, color=MLGREEN, linewidth=2, linestyle='--')

ax1.fill_between(x_line, y_line, 2, alpha=0.2, color=MLRED)
ax1.fill_between(x_line, y_line, -1, alpha=0.2, color=MLBLUE)

ax1.set_title('AND Gate: Solvable!', fontsize=11, fontweight='bold', color=MLGREEN)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.grid(alpha=0.3)

# Truth table
ax1.text(0.7, -0.3, 'AND: 1 only if both inputs are 1', fontsize=9)

# Plot 2: XOR gate (not solvable)
ax2 = axes[0, 1]

# XOR truth table
xor_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_labels = np.array([0, 1, 1, 0])

colors = [MLBLUE if l == 0 else MLRED for l in xor_labels]
ax2.scatter(xor_points[:, 0], xor_points[:, 1], c=colors, s=200, edgecolors='black', linewidths=2)

# Label points
for i, (x, y) in enumerate(xor_points):
    ax2.annotate(f'{int(xor_labels[i])}', (x, y), xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold')

# Try multiple lines (all fail)
ax2.plot([-0.5, 1.5], [0.5, 0.5], color='gray', linewidth=1, linestyle=':', alpha=0.5)
ax2.plot([0.5, 0.5], [-0.5, 1.5], color='gray', linewidth=1, linestyle=':', alpha=0.5)
ax2.plot([-0.5, 1.5], [1.5, -0.5], color='gray', linewidth=1, linestyle=':', alpha=0.5)

ax2.set_title('XOR Gate: No Single Line Works!', fontsize=11, fontweight='bold', color=MLRED)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)
ax2.grid(alpha=0.3)

# Truth table
ax2.text(0.3, -0.35, 'XOR: 1 if inputs differ', fontsize=9)

# Plot 3: The historical impact
ax3 = axes[1, 0]
ax3.axis('off')

history = '''
THE HISTORICAL IMPACT OF XOR

1969: MINSKY & PAPERT BOOK
--------------------------
"Perceptrons" proved that a single
perceptron cannot solve XOR.


THE CONSEQUENCES:
-----------------
- Funding for neural networks dried up
- Researchers abandoned the field
- "First AI Winter" (1970s-1980s)


WHY XOR IS HARD:
----------------
XOR Truth Table:
x1  x2  |  XOR
0   0   |   0
0   1   |   1
1   0   |   1
1   1   |   0

Diagonal points have same label!
No line can separate them.


WHAT THEY MISSED:
-----------------
Minsky & Papert knew the solution:
MULTI-LAYER networks CAN solve XOR!

But they argued:
"No one knows how to train them"

This was wrong - backpropagation
was already known (but forgotten).


LESSON:
-------
Never dismiss an idea because of
limitations in its simplest form.
'''

ax3.text(0.02, 0.98, history, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Historical Impact', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Solution with MLP
ax4 = axes[1, 1]
ax4.axis('off')

solution = '''
THE SOLUTION: MULTI-LAYER PERCEPTRON

XOR CAN BE BUILT FROM AND, OR, NOT:
-----------------------------------
XOR(x1, x2) = AND(OR(x1, x2), NOT(AND(x1, x2)))

Or equivalently:
XOR(x1, x2) = OR(AND(x1, NOT(x2)), AND(NOT(x1), x2))


2-LAYER NETWORK SOLUTION:
-------------------------
Hidden Layer (2 neurons):
  h1 = step(x1 + x2 - 0.5)      # OR-like
  h2 = step(-x1 - x2 + 1.5)     # NAND-like

Output Layer (1 neuron):
  y = step(h1 + h2 - 1.5)       # AND-like


GEOMETRIC INTERPRETATION:
-------------------------
Hidden layer creates TWO boundaries.
Output layer combines them.

Single perceptron: One line
Two-layer network: Any polygon region!


THE KEY INSIGHT:
----------------
Adding layers = adding complexity

One layer: Linear boundaries
Two layers: Convex regions
Three+ layers: ANY shape!

This is the foundation of deep learning.
'''

ax4.text(0.02, 0.98, solution, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('MLP Solution', fontsize=11, fontweight='bold', color=MLGREEN)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
