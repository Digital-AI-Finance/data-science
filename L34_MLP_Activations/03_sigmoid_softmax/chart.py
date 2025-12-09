"""Sigmoid and Softmax - Output Layer Activations"""
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
fig.suptitle('Output Activations: Sigmoid and Softmax', fontsize=14, fontweight='bold', color=MLPURPLE)

z = np.linspace(-5, 5, 200)

# Plot 1: Sigmoid for binary classification
ax1 = axes[0, 0]

sigmoid = 1 / (1 + np.exp(-z))
ax1.plot(z, sigmoid, color=MLBLUE, linewidth=3)

ax1.axhline(0.5, color='gray', linewidth=1, linestyle='--')
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axhline(1, color='black', linewidth=0.5, linestyle=':')
ax1.axvline(0, color='black', linewidth=0.5)

ax1.set_title('Sigmoid: Binary Classification Output', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('z (logit)')
ax1.set_ylabel('P(y=1)')
ax1.grid(alpha=0.3)

# Annotations
ax1.text(2, 0.3, r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))
ax1.text(-4, 0.8, 'Output: P(y=1)\nRange: [0, 1]', fontsize=9)

# Plot 2: Softmax visualization
ax2 = axes[0, 1]

# Softmax example with 3 classes
logits = np.array([2.0, 1.0, 0.5])
exp_logits = np.exp(logits)
softmax = exp_logits / exp_logits.sum()

colors = [MLBLUE, MLGREEN, MLORANGE]
bars = ax2.bar(['Class 0', 'Class 1', 'Class 2'], softmax, color=colors, edgecolor='black')

ax2.set_title('Softmax: Multi-Class Probabilities', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Probability')
ax2.set_ylim(0, 1)

# Add probability labels
for bar, prob in zip(bars, softmax):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{prob:.2f}', ha='center', fontsize=10, fontweight='bold')

# Add logits
for i, (bar, logit) in enumerate(zip(bars, logits)):
    ax2.text(bar.get_x() + bar.get_width()/2, 0.05,
             f'z={logit}', ha='center', fontsize=9, color='white')

ax2.text(1, 0.85, 'Sum = 1.00', fontsize=10, ha='center',
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: When to use what
ax3 = axes[1, 0]
ax3.axis('off')

when_to_use = '''
WHEN TO USE WHICH ACTIVATION

OUTPUT LAYER ACTIVATIONS:
-------------------------

SIGMOID:
- Binary classification (2 classes)
- Multi-label (each output independent)
- Output: probability for ONE class
- Loss: Binary cross-entropy

  model.add(Dense(1, activation='sigmoid'))


SOFTMAX:
- Multi-class classification (3+ classes)
- Classes are MUTUALLY EXCLUSIVE
- Output: probabilities for ALL classes
- Sum of outputs = 1
- Loss: Categorical cross-entropy

  model.add(Dense(num_classes, activation='softmax'))


LINEAR (no activation):
- Regression (predicting continuous values)
- Output: any real number

  model.add(Dense(1))  # Linear by default


SUMMARY TABLE:
--------------
Task                | Output | Activation
--------------------|--------|------------
Regression          |   1    | Linear
Binary class        |   1    | Sigmoid
Multi-class (3)     |   3    | Softmax
Multi-label (3)     |   3    | Sigmoid
'''

ax3.text(0.02, 0.98, when_to_use, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('When to Use What', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Softmax formula
ax4 = axes[1, 1]
ax4.axis('off')

softmax_math = '''
SOFTMAX FORMULA

FORMULA:
--------
softmax(z_i) = exp(z_i) / SUM_j exp(z_j)


EXAMPLE:
--------
Logits: z = [2.0, 1.0, 0.5]

exp(z) = [e^2.0, e^1.0, e^0.5]
       = [7.39, 2.72, 1.65]

Sum = 7.39 + 2.72 + 1.65 = 11.76

Softmax:
  P(class 0) = 7.39 / 11.76 = 0.63
  P(class 1) = 2.72 / 11.76 = 0.23
  P(class 2) = 1.65 / 11.76 = 0.14

Total = 1.00 (always!)


PROPERTIES:
-----------
1. Outputs always positive
2. Outputs sum to 1
3. Preserves order (larger z -> larger P)
4. Differentiable


NUMERICAL STABILITY:
-------------------
To avoid overflow, subtract max(z) first:
z = z - max(z)
Then apply softmax.

Libraries do this automatically.
'''

ax4.text(0.02, 0.98, softmax_math, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Softmax Formula', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
