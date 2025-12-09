"""Backpropagation Intuition - The Chain Rule in Action"""
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
fig.suptitle('Backpropagation: How Gradients Flow Backward', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: The chain rule
ax1 = axes[0, 0]
ax1.axis('off')

chain_rule = '''
THE CHAIN RULE

THE PROBLEM:
------------
How does changing W1 (first layer)
affect the final Loss?

W1 affects a1, which affects z2,
which affects a2, which affects Loss.


THE CHAIN RULE SOLUTION:
------------------------
If y = f(g(x)), then:

    dy/dx = dy/dg * dg/dx


APPLIED TO NEURAL NETWORKS:
---------------------------
Loss depends on: a2 -> z2 -> a1 -> z1 -> W1

dL/dW1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dW1


BACKPROPAGATION:
----------------
1. Compute dL/da2 at output
2. Propagate backward through activation: da2/dz2
3. Propagate through linear layer: dz2/da1
4. Continue backward to first layer
5. Update all weights


KEY INSIGHT:
------------
We compute gradients in REVERSE order:
- Forward: input -> hidden -> output
- Backward: output -> hidden -> input

This is why it's called "back" propagation!
'''

ax1.text(0.02, 0.98, chain_rule, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('The Chain Rule', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual backprop flow
ax2 = axes[0, 1]
ax2.axis('off')

# Draw network with backward arrows
layer_x = [0.15, 0.4, 0.65, 0.9]
layer_sizes = [2, 3, 3, 1]
layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']

# Draw nodes
colors = [MLBLUE, MLGREEN, MLGREEN, MLRED]
for l, (x, size, name, color) in enumerate(zip(layer_x, layer_sizes, layer_names, colors)):
    y_positions = np.linspace(0.3, 0.7, size)
    for i, y in enumerate(y_positions):
        circle = plt.Circle((x, y), 0.03, color=color, ec='black', linewidth=1.5)
        ax2.add_patch(circle)
    ax2.text(x, 0.15, name, ha='center', fontsize=9, fontweight='bold')

# Draw backward arrows (red, dashed)
for l in range(len(layer_x) - 1, 0, -1):
    mid_x = (layer_x[l] + layer_x[l-1]) / 2
    ax2.annotate('', xy=(layer_x[l-1] + 0.05, 0.5), xytext=(layer_x[l] - 0.05, 0.5),
                arrowprops=dict(arrowstyle='->', color=MLRED, lw=3, linestyle='--'))
    ax2.text(mid_x, 0.58, f'dL/dW{l}', ha='center', fontsize=8, color=MLRED)

# Add Loss at end
ax2.text(0.95, 0.5, 'L', fontsize=14, fontweight='bold', color=MLRED, ha='left', va='center')
ax2.annotate('', xy=(0.92, 0.5), xytext=(0.98, 0.5),
            arrowprops=dict(arrowstyle='->', color=MLRED, lw=2))

# Title
ax2.text(0.5, 0.85, 'Gradients Flow BACKWARD', fontsize=11, ha='center',
         color=MLRED, fontweight='bold')
ax2.text(0.5, 0.08, 'Start from Loss, propagate errors back to each layer',
         fontsize=9, ha='center')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Backpropagation Flow', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Step by step example
ax3 = axes[1, 0]
ax3.axis('off')

example = '''
BACKPROP STEP BY STEP

NETWORK: Input(2) -> Hidden(2) -> Output(1)

FORWARD PASS:
-------------
x = [1, 2]
z1 = W1 @ x + b1 = [0.5, 0.8]
a1 = relu(z1) = [0.5, 0.8]
z2 = W2 @ a1 + b2 = [0.7]
y_hat = sigmoid(z2) = [0.67]

y_true = [1]
Loss = -(y*log(y_hat) + (1-y)*log(1-y_hat)) = 0.4


BACKWARD PASS:
--------------
Step 1: Output layer gradient
    dL/dz2 = y_hat - y = 0.67 - 1 = -0.33

Step 2: Gradient for W2
    dL/dW2 = dL/dz2 * a1^T
           = -0.33 * [0.5, 0.8]
           = [-0.165, -0.264]

Step 3: Propagate to hidden layer
    dL/da1 = W2^T * dL/dz2

Step 4: Through ReLU
    dL/dz1 = dL/da1 * relu'(z1)
           (relu' = 1 if z > 0, else 0)

Step 5: Gradient for W1
    dL/dW1 = dL/dz1 * x^T


UPDATE: W := W - lr * dL/dW
'''

ax3.text(0.02, 0.98, example, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Backprop Step by Step', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Key derivatives
ax4 = axes[1, 1]
ax4.axis('off')

derivatives = '''
KEY DERIVATIVES FOR BACKPROP

ACTIVATION DERIVATIVES:
-----------------------
Sigmoid:
    sigmoid(z) = 1/(1+e^(-z))
    sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))

ReLU:
    relu(z) = max(0, z)
    relu'(z) = 1 if z > 0
               0 if z <= 0

Softmax + Cross-Entropy:
    (combined derivative is simple!)
    dL/dz = y_hat - y_true


LOSS DERIVATIVES:
-----------------
MSE Loss:
    L = (y - y_hat)^2
    dL/dy_hat = 2(y_hat - y)

Binary Cross-Entropy:
    L = -[y*log(y_hat) + (1-y)*log(1-y_hat)]
    dL/dy_hat = (y_hat - y) / (y_hat * (1 - y_hat))


WEIGHT GRADIENT:
----------------
For any layer with input a and output z:
    z = W @ a + b

    dL/dW = dL/dz @ a^T
    dL/db = dL/dz
    dL/da = W^T @ dL/dz  (for next layer)


GOOD NEWS:
----------
Keras/TensorFlow computes all this automatically!
You just call model.fit() and it handles backprop.
'''

ax4.text(0.02, 0.98, derivatives, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Key Derivatives', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
