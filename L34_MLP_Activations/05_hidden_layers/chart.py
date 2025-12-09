"""Hidden Layers - Choosing Architecture"""
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
fig.suptitle('Hidden Layers: How Many and How Wide?', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Effect of depth
ax1 = axes[0, 0]

# Simulated performance vs depth
depths = [0, 1, 2, 3, 4, 5]
performance = [0.65, 0.78, 0.85, 0.88, 0.87, 0.85]
training_time = [1, 2, 4, 8, 16, 32]

ax1.plot(depths, performance, 'o-', color=MLBLUE, linewidth=2, markersize=8, label='Test Accuracy')

ax1_twin = ax1.twinx()
ax1_twin.plot(depths, training_time, 's--', color=MLRED, linewidth=2, markersize=8, label='Training Time')

ax1.set_xlabel('Number of Hidden Layers')
ax1.set_ylabel('Test Accuracy', color=MLBLUE)
ax1_twin.set_ylabel('Training Time (relative)', color=MLRED)

ax1.set_title('Effect of Network Depth', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.grid(alpha=0.3)

# Optimal region
ax1.axvspan(2, 3, alpha=0.2, color=MLGREEN)
ax1.text(2.5, 0.7, 'Sweet\nspot', ha='center', fontsize=9, color=MLGREEN)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)

# Plot 2: Effect of width
ax2 = axes[0, 1]

widths = [8, 16, 32, 64, 128, 256]
perf_width = [0.72, 0.80, 0.86, 0.89, 0.88, 0.87]
params = [w * 10 + w * w + w for w in widths]  # Rough param count

ax2.plot(widths, perf_width, 'o-', color=MLBLUE, linewidth=2, markersize=8, label='Test Accuracy')

ax2_twin = ax2.twinx()
ax2_twin.plot(widths, [p/1000 for p in params], 's--', color=MLORANGE, linewidth=2, markersize=8, label='Params (K)')

ax2.set_xlabel('Neurons per Hidden Layer')
ax2.set_ylabel('Test Accuracy', color=MLBLUE)
ax2_twin.set_ylabel('Parameters (thousands)', color=MLORANGE)

ax2.set_title('Effect of Layer Width', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3)

# Optimal region
ax2.axvspan(32, 64, alpha=0.2, color=MLGREEN)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)

# Plot 3: Guidelines
ax3 = axes[1, 0]
ax3.axis('off')

guidelines = '''
ARCHITECTURE GUIDELINES

NUMBER OF HIDDEN LAYERS:
------------------------
0 layers: Linear model (logistic/linear regression)
1 layer:  Most problems (universal approximator)
2 layers: Complex patterns
3+ layers: Very complex / large data

Start with 1-2 layers, add more if needed.


WIDTH (NEURONS PER LAYER):
--------------------------
Rule of thumb:
- Between input size and output size
- Powers of 2 (32, 64, 128, 256)
- Often funnel shape: larger -> smaller

Examples:
  (100 features, 10 classes)
  -> Dense(64) -> Dense(32) -> Dense(10)


FUNNEL vs CONSTANT:
-------------------
Funnel: 128 -> 64 -> 32
  Compresses information gradually

Constant: 64 -> 64 -> 64
  Works fine, simpler to tune


PRACTICAL ADVICE:
-----------------
1. Start simple (1 hidden layer, 64 neurons)
2. If underfitting: add layers/neurons
3. If overfitting: add regularization first
4. Tune architecture with validation data
'''

ax3.text(0.02, 0.98, guidelines, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Architecture Guidelines', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Example architectures
ax4 = axes[1, 1]
ax4.axis('off')

examples = '''
EXAMPLE ARCHITECTURES

TABULAR DATA (Finance):
-----------------------
Input: 20 features
Task: Binary classification

# Simple (start here)
Dense(32, 'relu')
Dense(16, 'relu')
Dense(1, 'sigmoid')

# If more complex
Dense(64, 'relu')
Dense(32, 'relu')
Dense(16, 'relu')
Dense(1, 'sigmoid')


LARGER DATASET:
---------------
Input: 100 features
Task: 10-class classification

Dense(128, 'relu')
Dense(64, 'relu')
Dense(32, 'relu')
Dense(10, 'softmax')


REGRESSION:
-----------
Input: 50 features
Task: Predict price

Dense(64, 'relu')
Dense(32, 'relu')
Dense(1)  # Linear output


TIPS:
-----
- More data -> more layers/neurons OK
- Always validate on held-out data
- Architecture is less important than:
  - Good data
  - Proper preprocessing
  - Regularization
'''

ax4.text(0.02, 0.98, examples, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Example Architectures', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
