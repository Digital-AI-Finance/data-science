"""Gradient Descent - The Optimization Algorithm"""
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
fig.suptitle('Gradient Descent: Finding Optimal Weights', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: 1D gradient descent visualization
ax1 = axes[0, 0]

# Loss function (parabola)
w = np.linspace(-3, 3, 100)
loss = w**2 + 0.5

ax1.plot(w, loss, color=MLBLUE, linewidth=2.5, label='Loss L(w)')

# Gradient descent path
w_path = [2.5]
lr = 0.3
for _ in range(8):
    gradient = 2 * w_path[-1]  # derivative of w^2
    w_new = w_path[-1] - lr * gradient
    w_path.append(w_new)

# Plot path
for i in range(len(w_path) - 1):
    ax1.scatter(w_path[i], w_path[i]**2 + 0.5, color=MLRED, s=80, zorder=5)
    ax1.annotate('', xy=(w_path[i+1], w_path[i+1]**2 + 0.5),
                xytext=(w_path[i], w_path[i]**2 + 0.5),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=1.5))

ax1.scatter(w_path[-1], w_path[-1]**2 + 0.5, color=MLGREEN, s=100, zorder=5, label='Final')
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

ax1.set_xlabel('Weight w')
ax1.set_ylabel('Loss L(w)')
ax1.set_title('1D Gradient Descent', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax1.text(1.5, 6, 'w_new = w - lr * gradient', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 2: 2D contour visualization
ax2 = axes[0, 1]

# 2D loss surface
w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)
W1, W2 = np.meshgrid(w1, w2)
Loss = W1**2 + W2**2 + 0.5*W1*W2 + 1

# Contour plot
contour = ax2.contour(W1, W2, Loss, levels=15, colors='gray', alpha=0.5)
ax2.contourf(W1, W2, Loss, levels=15, cmap='Blues', alpha=0.3)

# Gradient descent path in 2D
path_w1 = [2.5]
path_w2 = [2.0]
lr = 0.15

for _ in range(15):
    grad_w1 = 2*path_w1[-1] + 0.5*path_w2[-1]
    grad_w2 = 2*path_w2[-1] + 0.5*path_w1[-1]
    path_w1.append(path_w1[-1] - lr * grad_w1)
    path_w2.append(path_w2[-1] - lr * grad_w2)

ax2.plot(path_w1, path_w2, 'o-', color=MLRED, markersize=6, linewidth=2, label='GD path')
ax2.scatter(path_w1[0], path_w2[0], color=MLORANGE, s=100, zorder=5, label='Start')
ax2.scatter(path_w1[-1], path_w2[-1], color=MLGREEN, s=100, zorder=5, label='End')

ax2.set_xlabel('Weight w1')
ax2.set_ylabel('Weight w2')
ax2.set_title('2D Gradient Descent (Contour)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)

# Plot 3: Gradient descent algorithm
ax3 = axes[1, 0]
ax3.axis('off')

algorithm = '''
GRADIENT DESCENT ALGORITHM

THE IDEA:
---------
1. Start with random weights
2. Compute gradient (direction of steepest ascent)
3. Move in OPPOSITE direction (descent)
4. Repeat until convergence


UPDATE RULE:
------------
w_new = w_old - learning_rate * gradient

In math:
w := w - lr * dL/dw


FOR NEURAL NETWORKS:
--------------------
For each weight W_ij:

    W_ij := W_ij - lr * dL/dW_ij

The gradient dL/dW tells us:
- How much the loss changes when we change W
- Which direction makes the loss bigger/smaller


INTUITION:
----------
Imagine standing on a hill (loss surface):
- Gradient points UPHILL
- We want to go DOWNHILL
- So we move opposite to gradient
- Learning rate = step size


THE KEY QUESTION:
-----------------
How do we compute gradients for
weights deep inside the network?

Answer: BACKPROPAGATION
'''

ax3.text(0.02, 0.98, algorithm, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Gradient Descent Algorithm', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Variants comparison
ax4 = axes[1, 1]

# Generate noisy path for SGD
np.random.seed(42)
epochs = 50

# GD (smooth)
gd_loss = [10]
for i in range(epochs):
    gd_loss.append(gd_loss[-1] * 0.9)

# SGD (noisy)
sgd_loss = [10]
for i in range(epochs):
    sgd_loss.append(sgd_loss[-1] * 0.88 + np.random.randn() * 0.3)

# Mini-batch (moderate noise)
mb_loss = [10]
for i in range(epochs):
    mb_loss.append(mb_loss[-1] * 0.89 + np.random.randn() * 0.1)

ax4.plot(range(epochs+1), gd_loss, color=MLBLUE, linewidth=2, label='Batch GD')
ax4.plot(range(epochs+1), sgd_loss, color=MLRED, linewidth=1.5, alpha=0.7, label='SGD')
ax4.plot(range(epochs+1), mb_loss, color=MLGREEN, linewidth=2, label='Mini-batch')

ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.set_title('Gradient Descent Variants', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Add annotations
ax4.text(30, 4, 'Batch: Smooth but slow\nSGD: Noisy but fast\nMini-batch: Best of both',
         fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
