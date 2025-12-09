"""Decision Boundary - Where classification happens"""
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
fig.suptitle('Decision Boundaries in Logistic Regression', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: 1D decision boundary
ax1 = axes[0, 0]

# Generate 1D data
n = 100
x_class0 = np.random.normal(3, 1, n//2)
x_class1 = np.random.normal(7, 1, n//2)

ax1.scatter(x_class0, np.zeros(n//2) + 0.1 + np.random.uniform(-0.05, 0.05, n//2),
            c=MLBLUE, s=50, alpha=0.6, label='Class 0', edgecolors='black')
ax1.scatter(x_class1, np.zeros(n//2) - 0.1 + np.random.uniform(-0.05, 0.05, n//2),
            c=MLORANGE, s=50, alpha=0.6, label='Class 1', edgecolors='black')

# Decision boundary
ax1.axvline(5, color=MLRED, linewidth=3, linestyle='--', label='Decision Boundary')
ax1.fill_betweenx([-0.3, 0.3], 0, 5, alpha=0.1, color=MLBLUE)
ax1.fill_betweenx([-0.3, 0.3], 5, 10, alpha=0.1, color=MLORANGE)

ax1.set_title('1D Decision Boundary', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Feature x', fontsize=10)
ax1.set_xlim(0, 10)
ax1.set_ylim(-0.3, 0.3)
ax1.set_yticks([])
ax1.legend(fontsize=9)

# Plot 2: 2D decision boundary
ax2 = axes[0, 1]

# Generate 2D data
n = 150
X0 = np.random.multivariate_normal([3, 3], [[1, 0.5], [0.5, 1]], n//2)
X1 = np.random.multivariate_normal([6, 6], [[1, 0.5], [0.5, 1]], n//2)

ax2.scatter(X0[:, 0], X0[:, 1], c=MLBLUE, s=50, alpha=0.6, label='Class 0 (Down)', edgecolors='black')
ax2.scatter(X1[:, 0], X1[:, 1], c=MLORANGE, s=50, alpha=0.6, label='Class 1 (Up)', edgecolors='black')

# Decision boundary line: x2 = -b0/b1 - (b1/b2)*x1
# Simulated: x2 = 9 - x1
x_line = np.linspace(0, 9, 100)
y_line = 9 - x_line

ax2.plot(x_line, y_line, color=MLRED, linewidth=3, linestyle='--', label='Decision Boundary')
ax2.fill_between(x_line, 0, y_line, alpha=0.1, color=MLBLUE)
ax2.fill_between(x_line, y_line, 9, alpha=0.1, color=MLORANGE)

ax2.set_title('2D Decision Boundary (Linear)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature $x_1$ (Returns)', fontsize=10)
ax2.set_ylabel('Feature $x_2$ (Volume)', fontsize=10)
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 9)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Probability gradient
ax3 = axes[1, 0]

# Create meshgrid
xx, yy = np.meshgrid(np.linspace(0, 9, 100), np.linspace(0, 9, 100))
# Simulated probability: P = sigmoid(x1 + x2 - 9)
z = (xx + yy - 9)
P = 1 / (1 + np.exp(-z))

contour = ax3.contourf(xx, yy, P, levels=20, cmap='RdYlBu_r', alpha=0.8)
plt.colorbar(contour, ax=ax3, label='P(Class 1)')

# Decision boundary
ax3.contour(xx, yy, P, levels=[0.5], colors=[MLRED], linewidths=[3])

ax3.scatter(X0[:, 0], X0[:, 1], c='white', s=30, alpha=0.6, edgecolors=MLBLUE, linewidths=1.5)
ax3.scatter(X1[:, 0], X1[:, 1], c='white', s=30, alpha=0.6, edgecolors=MLORANGE, linewidths=1.5)

ax3.set_title('Probability Gradient with Decision Boundary', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Feature $x_1$', fontsize=10)
ax3.set_ylabel('Feature $x_2$', fontsize=10)

# Plot 4: Math behind decision boundary
ax4 = axes[1, 1]
ax4.axis('off')

math_text = r'''
DECISION BOUNDARY MATHEMATICS

The decision boundary is where P(Y=1|X) = 0.5

For sigmoid: sigma(z) = 0.5 when z = 0

Therefore, decision boundary is:

  z = beta_0 + beta_1*x_1 + beta_2*x_2 + ... = 0

EXAMPLE (2D case):
------------------
Fitted model: z = -4.5 + 0.5*x_1 + 0.5*x_2

Decision boundary: 0 = -4.5 + 0.5*x_1 + 0.5*x_2

Rearranging: x_2 = 9 - x_1

This is a LINE in 2D space!

PREDICTIONS:
------------
If z > 0: P > 0.5, predict Class 1
If z < 0: P < 0.5, predict Class 0
If z = 0: P = 0.5, on the boundary

KEY INSIGHT:
------------
Logistic regression always produces LINEAR
decision boundaries in feature space.

For non-linear boundaries, need:
- Polynomial features
- Kernel methods
- Neural networks
'''

ax4.text(0.02, 0.98, math_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Decision Boundary Math', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
