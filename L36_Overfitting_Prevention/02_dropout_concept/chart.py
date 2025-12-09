"""Dropout Concept - Random Neuron Deactivation"""
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
fig.suptitle('Dropout: Random Neuron Deactivation for Regularization', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Network without dropout
ax1 = axes[0, 0]
ax1.axis('off')

# Draw full network
layer_x = [0.2, 0.5, 0.8]
layer_sizes = [4, 5, 2]

for l, (x, size) in enumerate(zip(layer_x, layer_sizes)):
    y_positions = np.linspace(0.2, 0.8, size)
    for i, y in enumerate(y_positions):
        circle = plt.Circle((x, y), 0.04, color=MLBLUE, ec='black', linewidth=1.5)
        ax1.add_patch(circle)

# Draw all connections
for l in range(len(layer_x) - 1):
    y1_positions = np.linspace(0.2, 0.8, layer_sizes[l])
    y2_positions = np.linspace(0.2, 0.8, layer_sizes[l+1])
    for y1 in y1_positions:
        for y2 in y2_positions:
            ax1.plot([layer_x[l]+0.04, layer_x[l+1]-0.04], [y1, y2],
                    color='gray', linewidth=0.8, alpha=0.5)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('Standard Network (No Dropout)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.text(0.5, 0.05, 'All neurons active, all connections used', ha='center', fontsize=9)

# Plot 2: Network with dropout
ax2 = axes[0, 1]
ax2.axis('off')

# Dropout mask (which neurons to drop)
np.random.seed(123)
dropout_masks = [
    [True, True, True, True],  # Input layer (usually no dropout)
    [True, False, True, True, False],  # Hidden layer (50% dropout)
    [True, True]  # Output layer (no dropout)
]

for l, (x, size) in enumerate(zip(layer_x, layer_sizes)):
    y_positions = np.linspace(0.2, 0.8, size)
    for i, y in enumerate(y_positions):
        if dropout_masks[l][i]:
            circle = plt.Circle((x, y), 0.04, color=MLGREEN, ec='black', linewidth=1.5)
        else:
            circle = plt.Circle((x, y), 0.04, color='lightgray', ec='gray', linewidth=1, linestyle='--')
        ax2.add_patch(circle)
        if not dropout_masks[l][i]:
            ax2.plot([x-0.03, x+0.03], [y-0.03, y+0.03], color=MLRED, linewidth=2)
            ax2.plot([x-0.03, x+0.03], [y+0.03, y-0.03], color=MLRED, linewidth=2)

# Draw only active connections
for l in range(len(layer_x) - 1):
    y1_positions = np.linspace(0.2, 0.8, layer_sizes[l])
    y2_positions = np.linspace(0.2, 0.8, layer_sizes[l+1])
    for i, y1 in enumerate(y1_positions):
        if dropout_masks[l][i]:
            for j, y2 in enumerate(y2_positions):
                if dropout_masks[l+1][j]:
                    ax2.plot([layer_x[l]+0.04, layer_x[l+1]-0.04], [y1, y2],
                            color=MLGREEN, linewidth=1, alpha=0.7)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('With Dropout (Training)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.text(0.5, 0.05, 'Random neurons dropped each batch', ha='center', fontsize=9)

# Plot 3: Dropout explanation
ax3 = axes[1, 0]
ax3.axis('off')

explanation = '''
HOW DROPOUT WORKS

DURING TRAINING:
----------------
1. For each training batch:
   - Randomly "drop" neurons with probability p
   - Dropped neurons output = 0
   - Remaining neurons scaled by 1/(1-p)

2. Different neurons dropped each batch
   -> Network can't rely on any single neuron
   -> Forces redundant representations


DURING INFERENCE:
-----------------
- ALL neurons active
- No dropping (use full network)
- Outputs already scaled correctly


TYPICAL DROPOUT RATES:
----------------------
Input layer:   0.0 - 0.2 (or none)
Hidden layers: 0.2 - 0.5
Output layer:  None (never)

Most common: 0.3 - 0.5


WHY IT WORKS:
-------------
1. Prevents co-adaptation
   - Neurons can't rely on specific other neurons
   - Each neuron must be useful on its own

2. Implicit ensemble
   - Each batch trains different "sub-network"
   - Final model = average of all sub-networks

3. Adds noise
   - Regularization effect
   - Similar to data augmentation


ANALOGY:
--------
Like training a team where random members
are absent each day - everyone learns all roles!
'''

ax3.text(0.02, 0.98, explanation, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('How Dropout Works', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Effect of dropout rate
ax4 = axes[1, 1]

dropout_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Simulated validation accuracy for different dropout rates
# Too low: overfitting, too high: underfitting
np.random.seed(42)
val_accuracy = [0.78, 0.82, 0.86, 0.89, 0.88, 0.85, 0.80, 0.72, 0.60]
train_accuracy = [0.99, 0.97, 0.94, 0.91, 0.88, 0.84, 0.78, 0.70, 0.58]

ax4.plot(dropout_rates, train_accuracy, 'o-', color=MLBLUE, linewidth=2, markersize=8, label='Training Acc')
ax4.plot(dropout_rates, val_accuracy, 's-', color=MLORANGE, linewidth=2, markersize=8, label='Validation Acc')

# Mark optimal
optimal_idx = np.argmax(val_accuracy)
ax4.axvline(dropout_rates[optimal_idx], color=MLGREEN, linestyle='--', linewidth=2)
ax4.scatter([dropout_rates[optimal_idx]], [val_accuracy[optimal_idx]], color=MLGREEN, s=150, zorder=5)
ax4.text(dropout_rates[optimal_idx] + 0.03, 0.91, f'Optimal: {dropout_rates[optimal_idx]}', fontsize=9, color=MLGREEN)

ax4.fill_between([0, 0.2], 0.5, 1, alpha=0.1, color=MLRED)
ax4.fill_between([0.5, 0.8], 0.5, 1, alpha=0.1, color=MLBLUE)

ax4.text(0.1, 0.55, 'Too low\n(overfit)', ha='center', fontsize=8, color=MLRED)
ax4.text(0.65, 0.55, 'Too high\n(underfit)', ha='center', fontsize=8, color=MLBLUE)

ax4.set_xlabel('Dropout Rate')
ax4.set_ylabel('Accuracy')
ax4.set_title('Effect of Dropout Rate', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)
ax4.set_ylim(0.5, 1)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
