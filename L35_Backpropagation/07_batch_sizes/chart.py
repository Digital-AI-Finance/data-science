"""Batch Sizes - Trading Off Speed and Stability"""
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
fig.suptitle('Batch Sizes: Speed vs Stability Trade-off', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Effect of batch size on loss curves
ax1 = axes[0, 0]

epochs = np.arange(100)

# Batch size 1 (SGD) - very noisy
loss_bs1 = 1.5 * np.exp(-0.03 * epochs) + 0.2 + np.random.randn(100) * 0.15

# Batch size 32 - moderate noise
loss_bs32 = 1.5 * np.exp(-0.035 * epochs) + 0.18 + np.random.randn(100) * 0.04

# Batch size 128 - smooth
loss_bs128 = 1.5 * np.exp(-0.03 * epochs) + 0.17 + np.random.randn(100) * 0.015

# Full batch - very smooth but slower
loss_full = 1.5 * np.exp(-0.025 * epochs) + 0.16 + np.random.randn(100) * 0.005

ax1.plot(epochs, loss_bs1, color=MLRED, linewidth=1, alpha=0.6, label='Batch=1 (SGD)')
ax1.plot(epochs, loss_bs32, color=MLORANGE, linewidth=1.5, label='Batch=32')
ax1.plot(epochs, loss_bs128, color=MLGREEN, linewidth=2, label='Batch=128')
ax1.plot(epochs, loss_full, color=MLBLUE, linewidth=2, label='Full Batch')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves by Batch Size', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Gradient noise visualization
ax2 = axes[0, 1]

# Show gradient estimates with different batch sizes
x = np.arange(10)
true_gradient = np.ones(10) * -0.5

# Different noise levels
np.random.seed(42)
grad_bs1 = true_gradient + np.random.randn(10) * 0.8
grad_bs32 = true_gradient + np.random.randn(10) * 0.2
grad_bs128 = true_gradient + np.random.randn(10) * 0.08

ax2.axhline(-0.5, color='black', linewidth=2, linestyle='--', label='True Gradient')
ax2.scatter(x, grad_bs1, color=MLRED, s=80, alpha=0.6, label='Batch=1')
ax2.scatter(x+0.2, grad_bs32, color=MLORANGE, s=80, alpha=0.6, label='Batch=32')
ax2.scatter(x+0.4, grad_bs128, color=MLGREEN, s=80, alpha=0.6, label='Batch=128')

ax2.set_xlabel('Update Step')
ax2.set_ylabel('Gradient Estimate')
ax2.set_title('Gradient Noise by Batch Size', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax2.text(5, 0.3, 'Smaller batch = Noisier gradient\nBut: noise can help escape local minima!',
         fontsize=8, ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Batch size guidelines
ax3 = axes[1, 0]
ax3.axis('off')

guidelines = '''
BATCH SIZE GUIDELINES

DEFINITIONS:
------------
Batch Size = # samples before weight update
Epoch = One pass through all training data
Iteration = One weight update

If dataset has 1000 samples and batch_size=100:
  -> 10 iterations per epoch


TRADE-OFFS:
-----------
Small batch (1-32):
  + Regularization effect (noise helps generalization)
  + Can escape local minima
  + Lower memory usage
  - Noisy gradients
  - Slower (more updates)

Large batch (128-512):
  + Stable gradients
  + Faster (GPU parallelism)
  + Reproducible
  - May converge to sharp minima (worse generalization)
  - High memory usage


TYPICAL VALUES:
---------------
Common: 32, 64, 128, 256
Default: 32 (Keras default)
GPU memory limited: as large as fits


PRACTICAL TIPS:
---------------
1. Start with 32 or 64
2. If training unstable -> increase batch size
3. If overfitting -> try smaller batch
4. Adjust learning rate with batch size:
   - Larger batch -> can use larger learning rate
   - Rule of thumb: lr scales with sqrt(batch_size)


KERAS:
------
model.fit(X_train, y_train,
          batch_size=32,      # Common default
          epochs=100)


MEMORY CONSTRAINTS:
-------------------
If batch_size=128 causes OOM (Out of Memory):
- Reduce to 64 or 32
- Or use gradient accumulation
'''

ax3.text(0.02, 0.98, guidelines, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Batch Size Guidelines', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Training time vs batch size
ax4 = axes[1, 1]

batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512]

# Simulated training times (relative)
# Small batches: many updates, overhead
# Large batches: few updates, GPU efficient
training_time = [100, 35, 22, 15, 12, 10, 9, 9.5]  # Relative time

# Final validation loss
val_loss = [0.15, 0.14, 0.135, 0.13, 0.135, 0.14, 0.15, 0.16]

ax4_twin = ax4.twinx()

bars = ax4.bar(range(len(batch_sizes)), training_time, color=MLBLUE, alpha=0.6, label='Training Time')
line = ax4_twin.plot(range(len(batch_sizes)), val_loss, 'o-', color=MLRED, linewidth=2, markersize=8, label='Val Loss')

ax4.set_xticks(range(len(batch_sizes)))
ax4.set_xticklabels(batch_sizes)
ax4.set_xlabel('Batch Size')
ax4.set_ylabel('Training Time (relative)', color=MLBLUE)
ax4_twin.set_ylabel('Validation Loss', color=MLRED)

ax4.set_title('Batch Size Trade-offs', fontsize=11, fontweight='bold', color=MLPURPLE)

# Highlight sweet spot
ax4.axvspan(2.5, 4.5, alpha=0.2, color=MLGREEN)
ax4.text(3.5, 80, 'Sweet spot:\n32-64', ha='center', fontsize=9, color=MLGREEN)

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
