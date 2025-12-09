"""Linear Decision Boundary - What Perceptron Can Learn"""
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
fig.suptitle('Perceptron Decision Boundary: Linear Separability', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Linearly separable data
ax1 = axes[0, 0]

# Generate linearly separable data
class0 = np.random.randn(30, 2) * 0.8 + [-2, -1]
class1 = np.random.randn(30, 2) * 0.8 + [2, 1]

ax1.scatter(class0[:, 0], class0[:, 1], c=MLBLUE, s=50, label='Class 0', edgecolors='black')
ax1.scatter(class1[:, 0], class1[:, 1], c=MLRED, s=50, label='Class 1', edgecolors='black')

# Decision boundary
x_line = np.linspace(-5, 5, 100)
y_line = -x_line  # w1*x1 + w2*x2 = 0 with w=[1,1], b=0
ax1.plot(x_line, y_line, color=MLGREEN, linewidth=2, linestyle='--', label='Decision boundary')

ax1.fill_between(x_line, y_line, 5, alpha=0.1, color=MLRED)
ax1.fill_between(x_line, y_line, -5, alpha=0.1, color=MLBLUE)

ax1.set_title('Linearly Separable: Perceptron Works!', fontsize=11, fontweight='bold', color=MLGREEN)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)

# Plot 2: Non-linearly separable
ax2 = axes[0, 1]

# Generate non-linearly separable data
theta = np.linspace(0, 2*np.pi, 30)
class0_x = 1.5 * np.cos(theta) + np.random.randn(30) * 0.2
class0_y = 1.5 * np.sin(theta) + np.random.randn(30) * 0.2

class1_x = np.random.randn(30) * 0.5
class1_y = np.random.randn(30) * 0.5

ax2.scatter(class0_x, class0_y, c=MLBLUE, s=50, label='Class 0 (outer)', edgecolors='black')
ax2.scatter(class1_x, class1_y, c=MLRED, s=50, label='Class 1 (inner)', edgecolors='black')

# Try to draw a line (any line fails)
ax2.plot([-3, 3], [-2, 2], color=MLGREEN, linewidth=2, linestyle='--', label='Any line?')

ax2.set_title('NOT Linearly Separable: Perceptron Fails!', fontsize=11, fontweight='bold', color=MLRED)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)

# Plot 3: Decision boundary equation
ax3 = axes[1, 0]
ax3.axis('off')

equation = '''
DECISION BOUNDARY EQUATION

PERCEPTRON OUTPUT:
------------------
y = step(w1*x1 + w2*x2 + b)


DECISION BOUNDARY:
------------------
Where output changes from 0 to 1:
w1*x1 + w2*x2 + b = 0


SOLVING FOR LINE:
-----------------
x2 = -(w1/w2)*x1 - (b/w2)

This is a LINE in 2D!
- Slope = -w1/w2
- Intercept = -b/w2


EXAMPLE:
--------
w = [2, -1], b = 1

Boundary: 2*x1 - x2 + 1 = 0
          x2 = 2*x1 + 1

Points above line: 2*x1 - x2 + 1 < 0 -> class 0
Points below line: 2*x1 - x2 + 1 > 0 -> class 1


IN HIGHER DIMENSIONS:
---------------------
With n features:
w1*x1 + w2*x2 + ... + wn*xn + b = 0

This defines a HYPERPLANE
(generalizes line to n dimensions)
'''

ax3.text(0.02, 0.98, equation, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Decision Boundary Math', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Linearly separable definition
ax4 = axes[1, 1]
ax4.axis('off')

separable = '''
LINEAR SEPARABILITY

DEFINITION:
-----------
Data is LINEARLY SEPARABLE if there exists
a hyperplane that perfectly separates the classes.


WHAT PERCEPTRON CAN LEARN:
--------------------------
+ AND gate
+ OR gate
+ Simple classification boundaries

- XOR gate (next slide!)
- Circular patterns
- Any non-linear boundary


THE LIMITATION:
---------------
A single perceptron can only learn
LINEAR decision boundaries.

This is a fundamental limitation!


SOLUTION PREVIEW:
-----------------
Stack multiple perceptrons -> MLP
(Multi-Layer Perceptron)

MLPs can learn ANY decision boundary!
(Universal approximation theorem)


KEY TAKEAWAY:
-------------
Perceptron = powerful but limited
Deep networks = stack of perceptrons
              = unlimited power
'''

ax4.text(0.02, 0.98, separable, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Linear Separability', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
