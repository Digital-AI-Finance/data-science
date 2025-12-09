"""Sigmoid Function - The logistic function"""
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
fig.suptitle('The Sigmoid Function: Foundation of Logistic Regression', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic sigmoid function
ax1 = axes[0, 0]

z = np.linspace(-8, 8, 200)
sigma = 1 / (1 + np.exp(-z))

ax1.plot(z, sigma, color=MLBLUE, linewidth=3)
ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axhline(0, color='gray', linewidth=0.5)
ax1.axhline(1, color='gray', linewidth=0.5)
ax1.axvline(0, color='gray', linewidth=0.5)

# Add annotations
ax1.annotate('P = 0.5 at z = 0', xy=(0, 0.5), xytext=(2, 0.6),
             fontsize=10, arrowprops=dict(arrowstyle='->', color=MLORANGE))
ax1.annotate('Approaches 1', xy=(6, 0.98), fontsize=10, color=MLGREEN)
ax1.annotate('Approaches 0', xy=(-6, 0.02), fontsize=10, color=MLRED)

ax1.set_title('Sigmoid Function: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('z (linear combination)', fontsize=10)
ax1.set_ylabel('$\\sigma(z)$ (probability)', fontsize=10)
ax1.set_ylim(-0.1, 1.1)
ax1.grid(alpha=0.3)

# Plot 2: Why sigmoid?
ax2 = axes[0, 1]
ax2.axis('off')

why_sigmoid = r'''
WHY THE SIGMOID FUNCTION?

1. BOUNDED OUTPUT (0 to 1)
   - Perfect for probabilities!
   - P(Y=1|X) = sigma(z)

2. SMOOTH AND DIFFERENTIABLE
   - Allows gradient descent
   - Derivative: sigma'(z) = sigma(z) * (1 - sigma(z))

3. THE LINK TO LINEAR MODELS

   Linear model:
   z = beta_0 + beta_1*x_1 + ... + beta_p*x_p

   Logistic regression:
   P(Y=1|X) = sigma(z) = 1 / (1 + e^(-z))

4. LOG-ODDS INTERPRETATION

   log(P / (1-P)) = z

   - log-odds (logit) is linear in X
   - Coefficients = change in log-odds

5. DECISION BOUNDARY

   P = 0.5 when z = 0
   beta_0 + beta_1*x = 0  defines boundary
'''

ax2.text(0.02, 0.98, why_sigmoid, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Why Use Sigmoid?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Effect of coefficient magnitude
ax3 = axes[1, 0]

z = np.linspace(-5, 5, 200)

for beta, color, label in [(0.5, MLBLUE, 'beta=0.5 (gradual)'),
                            (1.0, MLORANGE, 'beta=1.0 (standard)'),
                            (2.0, MLGREEN, 'beta=2.0 (steep)'),
                            (5.0, MLRED, 'beta=5.0 (sharp)')]:
    sigma = 1 / (1 + np.exp(-beta * z))
    ax3.plot(z, sigma, color=color, linewidth=2.5, label=label)

ax3.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax3.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax3.set_title('Effect of Coefficient Magnitude on Decision Boundary', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('x (feature)', fontsize=10)
ax3.set_ylabel('P(Y=1|X)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: From linear to probability
ax4 = axes[1, 1]

# Generate sample data
x = np.linspace(0, 10, 200)
z = -5 + 1.0 * x
p = 1 / (1 + np.exp(-z))

# Linear part
ax4.plot(x, z, color=MLBLUE, linewidth=2, linestyle='--', label='Linear: z = -5 + x')

# Probability
ax4_twin = ax4.twinx()
ax4_twin.plot(x, p, color=MLRED, linewidth=2.5, label='P(Y=1) = sigmoid(z)')
ax4_twin.axhline(0.5, color='gray', linestyle=':', alpha=0.7)

# Decision threshold
decision_x = 5  # where z=0
ax4_twin.axvline(decision_x, color=MLGREEN, linewidth=2, linestyle='--', label='Decision boundary')

ax4.set_title('Linear Combination to Probability', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('x (feature)', fontsize=10)
ax4.set_ylabel('z (linear combination)', fontsize=10, color=MLBLUE)
ax4_twin.set_ylabel('P(Y=1|X)', fontsize=10, color=MLRED)

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
