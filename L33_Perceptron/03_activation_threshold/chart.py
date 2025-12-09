"""Activation Functions - From Step to Sigmoid"""
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
fig.suptitle('Activation Functions: From Step to Smooth', fontsize=14, fontweight='bold', color=MLPURPLE)

z = np.linspace(-5, 5, 200)

# Plot 1: Step function (original perceptron)
ax1 = axes[0, 0]

step = np.where(z >= 0, 1, 0)
ax1.plot(z, step, color=MLBLUE, linewidth=3, label='Step function')

ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)

ax1.set_title('Step Function (Original Perceptron)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('z')
ax1.set_ylabel('output')
ax1.grid(alpha=0.3)
ax1.set_ylim(-0.2, 1.2)

# Annotate
ax1.text(2, 0.6, 'step(z) = 1 if z >= 0\n         = 0 otherwise', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 2: Sigmoid function
ax2 = axes[0, 1]

sigmoid = 1 / (1 + np.exp(-z))
ax2.plot(z, sigmoid, color=MLRED, linewidth=3, label='Sigmoid')

ax2.axhline(0, color='black', linewidth=0.5)
ax2.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')
ax2.axvline(0, color='black', linewidth=0.5)

ax2.set_title('Sigmoid Function (Smooth Version)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('z')
ax2.set_ylabel('output')
ax2.grid(alpha=0.3)
ax2.set_ylim(-0.1, 1.1)

# Annotate
ax2.text(1, 0.3, r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Comparison
ax3 = axes[1, 0]

ax3.plot(z, step, color=MLBLUE, linewidth=2, linestyle='--', label='Step', alpha=0.7)
ax3.plot(z, sigmoid, color=MLRED, linewidth=3, label='Sigmoid')

ax3.axhline(0, color='black', linewidth=0.5)
ax3.axhline(0.5, color='gray', linewidth=0.5, linestyle=':')
ax3.axvline(0, color='black', linewidth=0.5)

ax3.set_title('Step vs Sigmoid Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('z')
ax3.set_ylabel('output')
ax3.legend()
ax3.grid(alpha=0.3)

# Annotate threshold
ax3.annotate('Threshold\nat z=0', xy=(0, 0.5), xytext=(1.5, 0.7),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

# Plot 4: Why sigmoid?
ax4 = axes[1, 1]
ax4.axis('off')

why = '''
WHY SIGMOID INSTEAD OF STEP?

STEP FUNCTION PROBLEM:
----------------------
- Not differentiable at z=0
- Gradient is 0 everywhere else
- Cannot use gradient descent!


SIGMOID ADVANTAGES:
-------------------
1. SMOOTH and differentiable everywhere
   d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))

2. OUTPUT RANGE [0, 1]
   Can be interpreted as probability

3. GRADIENT SIGNAL
   Always has non-zero gradient (except extremes)
   Allows learning via backpropagation


SIGMOID FORMULA:
----------------
sigma(z) = 1 / (1 + exp(-z))


PROPERTIES:
-----------
- sigma(0) = 0.5
- sigma(-inf) -> 0
- sigma(+inf) -> 1
- Symmetric: sigma(-z) = 1 - sigma(z)


LOGISTIC REGRESSION CONNECTION:
-------------------------------
Logistic regression IS a single-layer
neural network with sigmoid activation!

P(y=1|x) = sigmoid(w . x + b)
'''

ax4.text(0.02, 0.98, why, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Why Sigmoid?', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
