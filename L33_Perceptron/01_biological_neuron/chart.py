"""Biological Neuron - Neural Network Inspiration"""
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
fig.suptitle('Biological Neurons: The Inspiration for Neural Networks', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Biological neuron description
ax1 = axes[0, 0]
ax1.axis('off')

bio = '''
THE BIOLOGICAL NEURON

STRUCTURE:
----------
DENDRITES: Receive signals from other neurons
           (inputs)

CELL BODY: Processes incoming signals
           Sums up all inputs

AXON: Transmits output signal to other neurons
      (output)

SYNAPSE: Connection point between neurons
         Has a "strength" (weight)


HOW IT WORKS:
-------------
1. Dendrites receive electrical signals
2. Cell body sums up all signals
3. If sum exceeds THRESHOLD -> fires!
4. Signal travels down axon to next neuron


SIMPLIFIED MODEL:
-----------------
- Inputs: x1, x2, ..., xn
- Weights: w1, w2, ..., wn
- Sum: z = w1*x1 + w2*x2 + ... + wn*xn
- Activation: if z > threshold: output = 1
              else: output = 0


This is the PERCEPTRON!
'''

ax1.text(0.02, 0.98, bio, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Biological Neuron', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Simplified neuron diagram
ax2 = axes[0, 1]
ax2.axis('off')

# Draw neuron components
# Dendrites (inputs)
for i, y in enumerate([0.7, 0.5, 0.3]):
    ax2.annotate('', xy=(0.3, 0.5), xytext=(0.1, y),
                 arrowprops=dict(arrowstyle='->', color=MLBLUE, lw=2))
    ax2.text(0.05, y, f'x{i+1}', fontsize=12, fontweight='bold', color=MLBLUE, va='center')

# Cell body (circle)
circle = plt.Circle((0.4, 0.5), 0.1, color=MLGREEN, ec='black', linewidth=2)
ax2.add_patch(circle)
ax2.text(0.4, 0.5, 'Sum', fontsize=10, fontweight='bold', ha='center', va='center')

# Axon (output)
ax2.annotate('', xy=(0.7, 0.5), xytext=(0.5, 0.5),
             arrowprops=dict(arrowstyle='->', color=MLRED, lw=3))

# Activation function
bbox = dict(boxstyle='round', facecolor=MLORANGE, alpha=0.8)
ax2.text(0.65, 0.5, 'f()', fontsize=10, fontweight='bold', ha='center', va='center', bbox=bbox)

# Output
ax2.annotate('', xy=(0.9, 0.5), xytext=(0.75, 0.5),
             arrowprops=dict(arrowstyle='->', color=MLRED, lw=3))
ax2.text(0.95, 0.5, 'y', fontsize=12, fontweight='bold', color=MLRED, va='center')

# Labels
ax2.text(0.2, 0.85, 'Inputs\n(dendrites)', fontsize=9, ha='center', color=MLBLUE)
ax2.text(0.4, 0.3, 'Cell body', fontsize=9, ha='center', color=MLGREEN)
ax2.text(0.65, 0.35, 'Activation', fontsize=9, ha='center', color=MLORANGE)
ax2.text(0.9, 0.35, 'Output\n(axon)', fontsize=9, ha='center', color=MLRED)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Simplified Neuron Model', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Brain facts
ax3 = axes[1, 0]
ax3.axis('off')

facts = '''
BRAIN vs ARTIFICIAL NEURAL NETWORKS

HUMAN BRAIN:
------------
- ~86 billion neurons
- ~100 trillion synapses (connections)
- Parallel processing
- Low power (~20 watts)
- Learns from few examples
- General intelligence


ARTIFICIAL NEURAL NETWORKS:
---------------------------
- Millions to billions of "neurons"
- Sequential/parallel processing
- High power consumption
- Needs lots of data
- Task-specific


WHAT WE BORROWED:
-----------------
1. Weighted connections
2. Summation of inputs
3. Thresholding (activation)
4. Layered structure
5. Learning by adjusting weights


WHAT'S DIFFERENT:
-----------------
1. Biological neurons are much more complex
2. Brain has no backpropagation
3. Timing matters in biology
4. Brain is not organized in layers
5. Biological learning is local, not global


BUT THE INSPIRATION WORKED!
'''

ax3.text(0.02, 0.98, facts, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Brain vs ANNs', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Timeline
ax4 = axes[1, 1]
ax4.axis('off')

timeline = '''
HISTORY OF NEURAL NETWORKS

1943 - McCulloch & Pitts
       First mathematical model of a neuron

1958 - Frank Rosenblatt
       Perceptron: first trainable neural network

1969 - Minsky & Papert
       "Perceptrons" book: showed limitations
       -> First AI winter

1986 - Rumelhart, Hinton & Williams
       Backpropagation: training deep networks

1998 - Yann LeCun
       LeNet: first successful CNN

2012 - AlexNet
       ImageNet breakthrough -> Deep learning era

2017 - Transformers
       "Attention is All You Need"
       -> ChatGPT, modern LLMs


KEY INSIGHT:
------------
The perceptron (1958) is still the
fundamental building block of all
modern neural networks.

Understanding the perceptron =
Understanding deep learning foundations.
'''

ax4.text(0.02, 0.98, timeline, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Neural Network History', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
