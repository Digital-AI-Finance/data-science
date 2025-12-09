"""Forward Pass - How Information Flows Through the Network"""
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
fig.suptitle('Forward Pass: Computing Network Output', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Forward pass concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
FORWARD PASS (FORWARD PROPAGATION)

DEFINITION:
-----------
The process of computing the output of a
neural network given an input, by passing
data forward through each layer.


STEPS:
------
1. Input data enters the network
2. Each layer computes: output = activation(weights @ input + bias)
3. Output flows to next layer
4. Final layer produces prediction


DATA FLOW:
----------
Input -> Hidden Layer 1 -> Hidden Layer 2 -> ... -> Output

x -> z1 = W1x + b1 -> a1 = relu(z1)
  -> z2 = W2a1 + b2 -> a2 = relu(z2)
  -> z3 = W3a2 + b3 -> y_hat = sigmoid(z3)


KEY POINTS:
-----------
- No learning happens during forward pass
- Just computing predictions
- Must happen before backward pass
- Same process for training and inference
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Forward Pass Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual diagram of forward pass
ax2 = axes[0, 1]
ax2.axis('off')

# Draw network with forward arrows
layer_x = [0.15, 0.4, 0.65, 0.9]
layer_sizes = [3, 4, 4, 2]
layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
colors = [MLBLUE, MLGREEN, MLGREEN, MLRED]

# Draw nodes
for l, (x, size, name, color) in enumerate(zip(layer_x, layer_sizes, layer_names, colors)):
    y_positions = np.linspace(0.2, 0.8, size)
    for i, y in enumerate(y_positions):
        circle = plt.Circle((x, y), 0.03, color=color, ec='black', linewidth=1.5)
        ax2.add_patch(circle)
    ax2.text(x, 0.05, name, ha='center', fontsize=9, fontweight='bold')

# Draw forward arrows
for l in range(len(layer_x) - 1):
    mid_x = (layer_x[l] + layer_x[l+1]) / 2
    ax2.annotate('', xy=(layer_x[l+1] - 0.05, 0.5), xytext=(layer_x[l] + 0.05, 0.5),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=3))
    ax2.text(mid_x, 0.55, f'W{l+1}, b{l+1}', ha='center', fontsize=8, color=MLORANGE)

# Add computation labels
ax2.text(0.15, 0.9, 'x', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
ax2.text(0.4, 0.9, 'a1=relu(z1)', ha='center', fontsize=8, color=MLGREEN)
ax2.text(0.65, 0.9, 'a2=relu(z2)', ha='center', fontsize=8, color=MLGREEN)
ax2.text(0.9, 0.9, 'y_hat', ha='center', fontsize=10, fontweight='bold', color=MLRED)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Forward Pass Visualization', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Numerical example
ax3 = axes[1, 0]
ax3.axis('off')

example = '''
NUMERICAL EXAMPLE

INPUT: x = [0.5, 0.3]

LAYER 1:
--------
W1 = [[0.2, 0.4],      b1 = [0.1, 0.2]
      [0.3, 0.1]]

z1 = W1 @ x + b1
   = [[0.2*0.5 + 0.4*0.3],  + [0.1,
      [0.3*0.5 + 0.1*0.3]]    0.2]
   = [0.22, 0.18] + [0.1, 0.2]
   = [0.32, 0.38]

a1 = relu(z1) = [0.32, 0.38]  (both positive)


LAYER 2 (Output):
-----------------
W2 = [[0.5, 0.6]]      b2 = [0.1]

z2 = W2 @ a1 + b2
   = [0.5*0.32 + 0.6*0.38] + [0.1]
   = [0.388] + [0.1]
   = [0.488]

y_hat = sigmoid(z2) = sigmoid(0.488) = 0.62


PREDICTION: 0.62 (62% probability for class 1)
'''

ax3.text(0.02, 0.98, example, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Numerical Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Forward pass in code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
FORWARD PASS IN CODE

import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_pass(x, weights, biases):
    """
    Compute forward pass through network.

    Args:
        x: Input data (n_features,)
        weights: List of weight matrices
        biases: List of bias vectors

    Returns:
        activations: List of all layer activations
    """
    activations = [x]
    a = x

    for i, (W, b) in enumerate(zip(weights, biases)):
        # Linear transformation
        z = W @ a + b

        # Activation (ReLU for hidden, sigmoid for output)
        if i < len(weights) - 1:
            a = relu(z)
        else:
            a = sigmoid(z)

        activations.append(a)

    return activations

# Example usage
x = np.array([0.5, 0.3])
W1 = np.array([[0.2, 0.4], [0.3, 0.1]])
W2 = np.array([[0.5, 0.6]])
b1 = np.array([0.1, 0.2])
b2 = np.array([0.1])

activations = forward_pass(x, [W1, W2], [b1, b2])
print(f"Output: {activations[-1]}")  # 0.62
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Forward Pass Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
