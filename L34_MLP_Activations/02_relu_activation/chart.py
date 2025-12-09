"""ReLU Activation Function"""
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
fig.suptitle('ReLU: The Most Popular Activation Function', fontsize=14, fontweight='bold', color=MLPURPLE)

z = np.linspace(-5, 5, 200)

# Plot 1: ReLU function
ax1 = axes[0, 0]

relu = np.maximum(0, z)
ax1.plot(z, relu, color=MLBLUE, linewidth=3, label='ReLU')

ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)

ax1.set_title('ReLU: Rectified Linear Unit', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('z')
ax1.set_ylabel('ReLU(z)')
ax1.grid(alpha=0.3)
ax1.set_ylim(-1, 5)

# Formula
ax1.text(2, 3.5, 'ReLU(z) = max(0, z)', fontsize=11,
         bbox=dict(facecolor='white', alpha=0.8))
ax1.text(-4, 2, 'z < 0: output = 0\nz > 0: output = z', fontsize=10)

# Plot 2: ReLU vs Sigmoid
ax2 = axes[0, 1]

sigmoid = 1 / (1 + np.exp(-z))
ax2.plot(z, relu/5, color=MLBLUE, linewidth=2.5, label='ReLU (scaled)')
ax2.plot(z, sigmoid, color=MLRED, linewidth=2.5, label='Sigmoid')

ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)

ax2.set_title('ReLU vs Sigmoid', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('z')
ax2.set_ylabel('output')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Why ReLU is better
ax3 = axes[1, 0]
ax3.axis('off')

why_relu = '''
WHY ReLU DOMINATES DEEP LEARNING

SIGMOID PROBLEMS:
-----------------
1. VANISHING GRADIENT
   For |z| > 4: gradient -> 0
   Deep networks can't learn!

2. NOT ZERO-CENTERED
   Outputs always positive [0,1]
   Slows down training

3. EXPENSIVE
   exp() is computationally costly


RELU ADVANTAGES:
----------------
1. NO VANISHING GRADIENT (for z > 0)
   Gradient = 1 for positive inputs
   Deep networks can train!

2. COMPUTATIONALLY CHEAP
   Just max(0, z) - no exp()

3. SPARSE ACTIVATION
   Many neurons output 0
   More efficient, acts like feature selection

4. BIOLOGICALLY PLAUSIBLE
   Neurons either fire or don't


RELU PROBLEMS:
--------------
"Dying ReLU": If z < 0 always, neuron is dead
Gradient = 0 for negative inputs forever


DEFAULT CHOICE:
---------------
Use ReLU for hidden layers in 99% of cases!
'''

ax3.text(0.02, 0.98, why_relu, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Why ReLU?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: ReLU variants
ax4 = axes[1, 1]

# ReLU variants
leaky_relu = np.where(z > 0, z, 0.1 * z)
elu = np.where(z > 0, z, np.exp(z) - 1)

ax4.plot(z, relu, color=MLBLUE, linewidth=2.5, label='ReLU')
ax4.plot(z, leaky_relu, color=MLGREEN, linewidth=2.5, label='Leaky ReLU (a=0.1)')
ax4.plot(z, elu, color=MLORANGE, linewidth=2.5, label='ELU')

ax4.axhline(0, color='black', linewidth=0.5)
ax4.axvline(0, color='black', linewidth=0.5)

ax4.set_title('ReLU Variants', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('z')
ax4.set_ylabel('output')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
ax4.set_ylim(-2, 5)

# Annotate
ax4.text(-4, 1, 'Leaky ReLU:\nAllows small gradient\nfor z < 0', fontsize=8,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
