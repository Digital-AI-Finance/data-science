"""Parameter Counting - Understanding Model Complexity"""
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
fig.suptitle('Parameter Counting: Understanding Model Size', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Parameter formula
ax1 = axes[0, 0]
ax1.axis('off')

formula = '''
COUNTING PARAMETERS

FOR EACH DENSE LAYER:
---------------------
Parameters = (input_size x output_size) + output_size
           = weights + biases


FORMULA:
--------
params = n_in * n_out + n_out
       = n_out * (n_in + 1)


EXAMPLE LAYER:
--------------
Dense(64, input_shape=(10,))

n_in = 10
n_out = 64

params = 10 * 64 + 64 = 704
       = 640 weights + 64 biases


EXAMPLE NETWORK:
----------------
Input: 10 features
Dense(64): 10*64 + 64 = 704
Dense(32): 64*32 + 32 = 2,080
Dense(1):  32*1 + 1   = 33
----------------------------
Total:                  2,817 parameters


WHY IT MATTERS:
---------------
- More params = more expressive
- More params = more data needed
- More params = slower training
- More params = risk of overfitting
'''

ax1.text(0.02, 0.98, formula, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Parameter Formula', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual parameter diagram
ax2 = axes[0, 1]
ax2.axis('off')

# Draw a simple 3->4 layer with labeled connections
# Input layer
for i, y in enumerate([0.75, 0.5, 0.25]):
    circle = plt.Circle((0.2, y), 0.04, color=MLBLUE, ec='black')
    ax2.add_patch(circle)
    ax2.text(0.08, y, f'x{i+1}', fontsize=10, va='center')

# Output layer
for j, y in enumerate([0.85, 0.65, 0.45, 0.25]):
    circle = plt.Circle((0.7, y), 0.04, color=MLRED, ec='black')
    ax2.add_patch(circle)
    ax2.text(0.82, y, f'h{j+1}', fontsize=10, va='center')

# Draw connections (weights)
for i, y1 in enumerate([0.75, 0.5, 0.25]):
    for j, y2 in enumerate([0.85, 0.65, 0.45, 0.25]):
        ax2.plot([0.24, 0.66], [y1, y2], 'gray', linewidth=0.8, alpha=0.6)

# Annotations
ax2.text(0.45, 0.95, '3 x 4 = 12 weights', fontsize=10, ha='center', color=MLORANGE, fontweight='bold')
ax2.text(0.7, 0.08, '+ 4 biases', fontsize=10, ha='center', color=MLGREEN, fontweight='bold')
ax2.text(0.45, 0.02, 'Total: 12 + 4 = 16 params', fontsize=11, ha='center', fontweight='bold')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Visual: Dense(4, input_shape=(3,))', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Model.summary() output
ax3 = axes[1, 0]
ax3.axis('off')

summary = '''
READING model.summary()

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

OUTPUT:
_________________________________________________________________
Layer (type)              Output Shape          Param #
=================================================================
dense (Dense)             (None, 64)            704
_________________________________________________________________
dense_1 (Dense)           (None, 32)            2080
_________________________________________________________________
dense_2 (Dense)           (None, 1)             33
=================================================================
Total params: 2,817
Trainable params: 2,817
Non-trainable params: 0
_________________________________________________________________


UNDERSTANDING OUTPUT:
---------------------
- (None, 64): Batch size unspecified, 64 outputs
- Param # = n_in * n_out + n_out
- Trainable: Updated during training
- Non-trainable: Fixed (e.g., BatchNorm stats)
'''

ax3.text(0.02, 0.98, summary, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('model.summary() Output', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Parameter scaling
ax4 = axes[1, 1]

# Show how params scale with architecture
architectures = ['(10,32,1)', '(10,64,1)', '(10,64,32,1)', '(10,128,64,1)', '(10,256,128,64,1)']
param_counts = [
    10*32+32 + 32*1+1,  # 353
    10*64+64 + 64*1+1,  # 705
    10*64+64 + 64*32+32 + 32*1+1,  # 2817
    10*128+128 + 128*64+64 + 64*1+1,  # 9537
    10*256+256 + 256*128+128 + 128*64+64 + 64*1+1  # 43713
]

colors = [MLBLUE, MLBLUE, MLGREEN, MLORANGE, MLRED]
bars = ax4.bar(range(len(architectures)), [p/1000 for p in param_counts], color=colors, edgecolor='black')

ax4.set_xticks(range(len(architectures)))
ax4.set_xticklabels(architectures, fontsize=8, rotation=15)
ax4.set_ylabel('Parameters (thousands)')
ax4.set_title('Parameter Scaling with Architecture', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3, axis='y')

# Add count labels
for bar, count in zip(bars, param_counts):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{count:,}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
