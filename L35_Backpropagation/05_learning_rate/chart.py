"""Learning Rate - The Most Important Hyperparameter"""
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
fig.suptitle('Learning Rate: The Most Important Hyperparameter', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Effect of different learning rates
ax1 = axes[0, 0]

# Simulate gradient descent with different learning rates
w = np.linspace(-3, 3, 100)
loss_surface = (w - 0.5)**2 + 0.5

ax1.plot(w, loss_surface, color='gray', linewidth=2, alpha=0.5)

# Too small lr
w_small = 2.5
path_small = [w_small]
lr_small = 0.05
for _ in range(20):
    grad = 2 * (path_small[-1] - 0.5)
    path_small.append(path_small[-1] - lr_small * grad)

# Good lr
w_good = 2.5
path_good = [w_good]
lr_good = 0.3
for _ in range(10):
    grad = 2 * (path_good[-1] - 0.5)
    path_good.append(path_good[-1] - lr_good * grad)

# Too large lr
w_large = 2.5
path_large = [w_large]
lr_large = 0.9
for _ in range(10):
    grad = 2 * (path_large[-1] - 0.5)
    path_large.append(path_large[-1] - lr_large * grad)

# Plot paths
for i, p in enumerate(path_small[:15]):
    ax1.scatter(p, (p-0.5)**2 + 0.5, color=MLBLUE, s=30, alpha=0.5 + 0.03*i)
ax1.plot([p for p in path_small[:15]], [(p-0.5)**2 + 0.5 for p in path_small[:15]],
         'o-', color=MLBLUE, markersize=4, linewidth=1, alpha=0.7, label=f'lr={lr_small} (too small)')

for i, p in enumerate(path_good[:8]):
    ax1.scatter(p, (p-0.5)**2 + 0.5, color=MLGREEN, s=40, alpha=0.5 + 0.06*i)
ax1.plot([p for p in path_good[:8]], [(p-0.5)**2 + 0.5 for p in path_good[:8]],
         'o-', color=MLGREEN, markersize=5, linewidth=1.5, label=f'lr={lr_good} (good)')

for i, p in enumerate(path_large[:6]):
    ax1.scatter(p, (p-0.5)**2 + 0.5, color=MLRED, s=40, alpha=0.5 + 0.08*i)
ax1.plot([p for p in path_large[:6]], [(p-0.5)**2 + 0.5 for p in path_large[:6]],
         'o-', color=MLRED, markersize=5, linewidth=1.5, label=f'lr={lr_large} (too large)')

ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Weight')
ax1.set_ylabel('Loss')
ax1.set_title('Effect of Learning Rate', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(alpha=0.3)

# Plot 2: Loss curves
ax2 = axes[0, 1]

epochs = np.arange(100)

# Too small - slow convergence
loss_small = 2 * np.exp(-0.01 * epochs) + 0.3 + np.random.randn(100) * 0.02

# Good - optimal convergence
loss_good = 2 * np.exp(-0.05 * epochs) + 0.1 + np.random.randn(100) * 0.01

# Too large - oscillating
loss_large = 0.8 + 0.5 * np.sin(0.3 * epochs) + np.random.randn(100) * 0.15

# Way too large - diverging
loss_huge = 0.5 + 0.02 * epochs + np.random.randn(100) * 0.1

ax2.plot(epochs, loss_small, color=MLBLUE, linewidth=2, label='lr=0.0001 (slow)')
ax2.plot(epochs, loss_good, color=MLGREEN, linewidth=2, label='lr=0.001 (good)')
ax2.plot(epochs, loss_large, color=MLORANGE, linewidth=2, label='lr=0.1 (oscillating)')
ax2.plot(epochs[:50], loss_huge[:50], color=MLRED, linewidth=2, label='lr=1.0 (diverging)')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Learning Rate Effect on Training', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 3)

# Plot 3: Learning rate guidelines
ax3 = axes[1, 0]
ax3.axis('off')

guidelines = '''
LEARNING RATE GUIDELINES

TYPICAL VALUES:
---------------
- Default starting point: 0.001 (1e-3)
- Range to try: 0.0001 to 0.01
- For Adam optimizer: 0.001 usually works


TOO SMALL (lr < 0.0001):
------------------------
- Very slow convergence
- May get stuck in local minima
- Training takes forever
- But: stable, won't diverge


TOO LARGE (lr > 0.1):
---------------------
- Loss oscillates wildly
- May diverge (loss -> infinity)
- Overshoots the minimum
- Training is unstable


FINDING GOOD LR:
----------------
1. Start with 0.001
2. If loss not decreasing -> increase
3. If loss oscillating -> decrease
4. Try: [0.0001, 0.001, 0.01]


LEARNING RATE SCHEDULES:
------------------------
- Start high, decay over time
- Allows fast initial progress
- Fine-tuning near optimum

Common schedules:
- Step decay: lr = lr * 0.1 every N epochs
- Exponential: lr = lr_0 * decay^epoch
- Cosine annealing: smooth decay


KERAS EXAMPLE:
--------------
# With Adam (default lr=0.001)
model.compile(optimizer='adam', loss='mse')

# Custom learning rate
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
'''

ax3.text(0.02, 0.98, guidelines, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Learning Rate Guidelines', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Learning rate finder visualization
ax4 = axes[1, 1]

# Simulate learning rate finder
lr_range = np.logspace(-5, 0, 100)
# Loss typically decreases then increases
loss_finder = 2 * np.exp(-1000 * lr_range) + 0.1 + 0.5 * lr_range**0.5 + np.random.randn(100) * 0.02

ax4.plot(lr_range, loss_finder, color=MLPURPLE, linewidth=2)
ax4.set_xscale('log')

# Mark optimal region
optimal_idx = np.argmin(loss_finder)
ax4.axvline(lr_range[optimal_idx], color=MLGREEN, linestyle='--', linewidth=2, label=f'Optimal ~{lr_range[optimal_idx]:.4f}')
ax4.scatter([lr_range[optimal_idx]], [loss_finder[optimal_idx]], color=MLGREEN, s=100, zorder=5)

# Annotate regions
ax4.axvspan(1e-5, 1e-4, alpha=0.2, color=MLBLUE)
ax4.axvspan(1e-4, 1e-2, alpha=0.2, color=MLGREEN)
ax4.axvspan(1e-1, 1, alpha=0.2, color=MLRED)

ax4.text(3e-5, 1.5, 'Too\nsmall', fontsize=8, ha='center', color=MLBLUE)
ax4.text(1e-3, 1.5, 'Good\nrange', fontsize=8, ha='center', color=MLGREEN)
ax4.text(3e-1, 1.5, 'Too\nlarge', fontsize=8, ha='center', color=MLRED)

ax4.set_xlabel('Learning Rate (log scale)')
ax4.set_ylabel('Loss')
ax4.set_title('Learning Rate Finder', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
