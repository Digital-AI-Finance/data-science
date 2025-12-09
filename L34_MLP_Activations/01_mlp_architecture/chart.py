"""Multi-Layer Perceptron Architecture"""
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
fig.suptitle('Multi-Layer Perceptron (MLP) Architecture', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: MLP concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
MULTI-LAYER PERCEPTRON (MLP)

WHAT IS IT?
-----------
Multiple layers of perceptrons stacked together.
Also called: Feedforward Neural Network


ARCHITECTURE:
-------------
Input Layer -> Hidden Layers -> Output Layer


TERMINOLOGY:
------------
- Input layer: Receives features (not counted)
- Hidden layer: Intermediate computation
- Output layer: Final prediction
- Depth: Number of hidden layers
- Width: Neurons per layer


EXAMPLE:
--------
Input: 10 features
Hidden 1: 64 neurons
Hidden 2: 32 neurons
Output: 1 neuron (regression) or
        3 neurons (3-class classification)

Notation: MLP(10, 64, 32, 1) or
          "2-hidden-layer network"


WHY MULTIPLE LAYERS?
--------------------
Single perceptron: Linear boundaries only
Multiple layers: ANY function! (Universal)

Each layer learns more abstract features.
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('MLP Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Network diagram
ax2 = axes[0, 1]
ax2.axis('off')

# Layer positions
layers = {
    'input': ([0.15]*4, [0.8, 0.6, 0.4, 0.2]),
    'hidden1': ([0.4]*5, [0.85, 0.675, 0.5, 0.325, 0.15]),
    'hidden2': ([0.65]*3, [0.7, 0.5, 0.3]),
    'output': ([0.9]*1, [0.5])
}

layer_colors = {
    'input': MLBLUE,
    'hidden1': MLGREEN,
    'hidden2': MLGREEN,
    'output': MLRED
}

# Draw connections first
for (l1_name, l1_pos), (l2_name, l2_pos) in [
    (('input', layers['input']), ('hidden1', layers['hidden1'])),
    (('hidden1', layers['hidden1']), ('hidden2', layers['hidden2'])),
    (('hidden2', layers['hidden2']), ('output', layers['output']))
]:
    for x1, y1 in zip(l1_pos[0], l1_pos[1]):
        for x2, y2 in zip(l2_pos[0], l2_pos[1]):
            ax2.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.5)

# Draw neurons
for layer_name, (x_pos, y_pos) in layers.items():
    color = layer_colors[layer_name]
    for x, y in zip(x_pos, y_pos):
        circle = plt.Circle((x, y), 0.04, color=color, ec='black', linewidth=1.5, zorder=5)
        ax2.add_patch(circle)

# Labels
ax2.text(0.15, 0.95, 'Input\nLayer', ha='center', fontsize=9, color=MLBLUE, fontweight='bold')
ax2.text(0.525, 0.95, 'Hidden Layers', ha='center', fontsize=9, color=MLGREEN, fontweight='bold')
ax2.text(0.9, 0.95, 'Output', ha='center', fontsize=9, color=MLRED, fontweight='bold')

# Annotations
ax2.text(0.15, 0.05, '4 inputs', ha='center', fontsize=8)
ax2.text(0.4, 0.05, '5 neurons', ha='center', fontsize=8)
ax2.text(0.65, 0.05, '3 neurons', ha='center', fontsize=8)
ax2.text(0.9, 0.05, '1 output', ha='center', fontsize=8)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('MLP Architecture: (4, 5, 3, 1)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Feedforward computation
ax3 = axes[1, 0]
ax3.axis('off')

computation = '''
FEEDFORWARD COMPUTATION

FOR EACH LAYER:
---------------
1. Weighted sum: z = W . a_prev + b
2. Activation: a = f(z)


LAYER BY LAYER:
---------------
Input: a0 = x (features)

Hidden 1:
  z1 = W1 . a0 + b1
  a1 = f(z1)

Hidden 2:
  z2 = W2 . a1 + b2
  a2 = f(z2)

Output:
  z3 = W3 . a2 + b3
  a3 = f_out(z3)  # Different activation!


MATRIX DIMENSIONS:
------------------
If layer has n_in inputs, n_out outputs:
  W shape: (n_out, n_in)
  b shape: (n_out,)
  z shape: (n_out,)
  a shape: (n_out,)


EXAMPLE (4, 5, 3, 1):
--------------------
W1: (5, 4)   b1: (5,)
W2: (3, 5)   b2: (3,)
W3: (1, 3)   b3: (1,)

Total params: 5*4+5 + 3*5+3 + 1*3+1 = 43
'''

ax3.text(0.02, 0.98, computation, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Feedforward Computation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Why hidden layers help
ax4 = axes[1, 1]
ax4.axis('off')

why_hidden = '''
WHY HIDDEN LAYERS HELP

SINGLE LAYER (Perceptron):
--------------------------
Can only learn LINEAR boundaries.
Output = f(W . x + b)


ONE HIDDEN LAYER:
-----------------
Can learn CONVEX regions.
Any continuous function! (with enough neurons)


MULTIPLE HIDDEN LAYERS:
-----------------------
Can learn HIERARCHICAL features.
Each layer builds on the previous.


ANALOGY - IMAGE RECOGNITION:
----------------------------
Layer 1: Edges, simple patterns
Layer 2: Shapes (circles, corners)
Layer 3: Parts (eyes, wheels)
Layer 4: Objects (faces, cars)


DEPTH vs WIDTH:
---------------
Deeper (more layers) often better than wider.
Deeper = more abstract features.
Deeper = exponentially more expressive.

BUT: Harder to train (vanishing gradients)
Solution: Better activations, batch norm, skip connections


MODERN NETWORKS:
----------------
- ResNet: 152 layers
- GPT-4: ~100+ layers
- BERT: 12-24 layers
'''

ax4.text(0.02, 0.98, why_hidden, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Why Hidden Layers Help', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
