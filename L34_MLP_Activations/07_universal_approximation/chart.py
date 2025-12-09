"""Universal Approximation Theorem"""
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
fig.suptitle('Universal Approximation Theorem', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: The theorem
ax1 = axes[0, 0]
ax1.axis('off')

theorem = '''
UNIVERSAL APPROXIMATION THEOREM

STATEMENT (Cybenko, 1989; Hornik, 1991):
----------------------------------------
A feedforward network with a single hidden layer
containing a finite number of neurons can
approximate any continuous function on compact
subsets of R^n, to any desired accuracy.


IN PLAIN ENGLISH:
-----------------
ONE hidden layer + enough neurons
= can learn ANY continuous function


REQUIREMENTS:
-------------
- Non-linear activation function (sigmoid, ReLU)
- "Enough" neurons in hidden layer
- Compact (bounded) input domain


WHAT IT MEANS:
--------------
Neural networks are theoretically powerful.
They CAN represent any function.

WHAT IT DOESN'T MEAN:
---------------------
- Easy to find the right weights (training)
- Efficient representation (may need millions of neurons)
- Will generalize well
- Practical for all problems


KEY INSIGHT:
------------
The theorem guarantees EXISTENCE of a solution,
not that we can FIND it or that it's EFFICIENT.
'''

ax1.text(0.02, 0.98, theorem, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('The Theorem', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Approximating a complex function
ax2 = axes[0, 1]

x = np.linspace(-2, 2, 200)

# Target function
y_true = np.sin(3*x) + 0.5*np.cos(7*x)

# Approximations with increasing neurons
def approx(x, n_neurons):
    np.random.seed(42)
    y = np.zeros_like(x)
    for _ in range(n_neurons):
        w = np.random.randn() * 2
        b = np.random.randn()
        y += np.random.randn() * np.maximum(0, w*x + b)  # ReLU
    return y / n_neurons * 2

ax2.plot(x, y_true, color='black', linewidth=2.5, label='Target function')
ax2.plot(x, approx(x, 5), color=MLRED, linewidth=1.5, alpha=0.7, label='5 neurons')
ax2.plot(x, approx(x, 20), color=MLORANGE, linewidth=1.5, alpha=0.7, label='20 neurons')

ax2.set_title('Function Approximation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax2.text(0, -2, 'More neurons = better approximation', fontsize=9, ha='center',
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Decision boundary complexity
ax3 = axes[1, 0]

# Generate XOR-like data
n_points = 100
x1 = np.concatenate([np.random.randn(n_points//4)*0.3 + 0, np.random.randn(n_points//4)*0.3 + 1,
                     np.random.randn(n_points//4)*0.3 + 0, np.random.randn(n_points//4)*0.3 + 1])
x2 = np.concatenate([np.random.randn(n_points//4)*0.3 + 0, np.random.randn(n_points//4)*0.3 + 0,
                     np.random.randn(n_points//4)*0.3 + 1, np.random.randn(n_points//4)*0.3 + 1])
y = np.array([0]*(n_points//4) + [1]*(n_points//4) + [1]*(n_points//4) + [0]*(n_points//4))

colors = [MLBLUE if label == 0 else MLRED for label in y]
ax3.scatter(x1, x2, c=colors, s=30, alpha=0.6, edgecolors='black', linewidths=0.3)

# Draw non-linear boundary (XOR solution)
x_bound = np.linspace(-0.5, 1.5, 100)
ax3.plot(x_bound, x_bound, 'k--', linewidth=1, alpha=0.5)
ax3.plot(x_bound, -x_bound + 1, 'k--', linewidth=1, alpha=0.5)

ax3.set_title('MLPs Can Learn Non-Linear Boundaries', fontsize=11, fontweight='bold', color=MLGREEN)
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.grid(alpha=0.3)

ax3.text(0.5, -0.3, 'XOR pattern: Not solvable by single perceptron\nBut MLP handles it easily!',
         fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Plot 4: Practical implications
ax4 = axes[1, 1]
ax4.axis('off')

practical = '''
PRACTICAL IMPLICATIONS

THEORY vs PRACTICE:
-------------------
Theorem: 1 layer can approximate anything
Practice: Deep networks work much better

WHY?
- Deeper = more efficient representation
- Deeper = hierarchical features
- Deeper = easier optimization (sometimes)


WIDTH vs DEPTH:
---------------
Wide + shallow: Can represent, but inefficient
Narrow + deep: More efficient, learns hierarchy


EXAMPLE - XOR:
--------------
Width approach: 4 neurons in one layer
Depth approach: 2 neurons in each of 2 layers

Both work, but depth generalizes better.


THE REAL BOTTLENECK:
--------------------
Not expressiveness (solved by theorem)
But:
1. Finding good weights (optimization)
2. Generalization (need data + regularization)
3. Computational cost


TAKEAWAY:
---------
Neural networks CAN learn anything.
The challenge is getting them to learn
the RIGHT thing efficiently.
'''

ax4.text(0.02, 0.98, practical, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Practical Implications', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
