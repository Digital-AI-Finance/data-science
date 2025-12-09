"""Binomial Distribution - Success/Failure trials"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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
fig.suptitle('The Binomial Distribution', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic binomial (coin flip example)
ax1 = axes[0, 0]
n, p = 20, 0.5
x = np.arange(0, n+1)
pmf = stats.binom.pmf(x, n, p)

ax1.bar(x, pmf, color=MLBLUE, alpha=0.7, edgecolor='black')
ax1.axvline(n*p, color=MLRED, linestyle='--', linewidth=2, label=f'Mean = n*p = {n*p:.0f}')

ax1.set_title(f'Binomial(n={n}, p={p}) - Fair Coin Flips', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Number of Heads', fontsize=10)
ax1.set_ylabel('Probability', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Different probabilities
ax2 = axes[0, 1]
n = 20
probs = [0.2, 0.5, 0.8]
colors = [MLRED, MLBLUE, MLGREEN]
x = np.arange(0, n+1)

for p, color in zip(probs, colors):
    pmf = stats.binom.pmf(x, n, p)
    ax2.plot(x, pmf, 'o-', color=color, linewidth=2, markersize=6, label=f'p = {p}')

ax2.set_title('Effect of Probability p', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Number of Successes', fontsize=10)
ax2.set_ylabel('Probability', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Finance application - Trading days positive
ax3 = axes[1, 0]
# Probability of profitable trading days
n = 252  # Trading days per year
p = 0.52  # Slightly better than random

# Probability of at least k profitable days
k = np.arange(100, 180)
prob_at_least_k = 1 - stats.binom.cdf(k-1, n, p)

ax3.plot(k, prob_at_least_k, color=MLBLUE, linewidth=2.5)
ax3.fill_between(k, prob_at_least_k, alpha=0.3, color=MLBLUE)

# Mark key points
ax3.axhline(0.5, color=MLRED, linestyle='--', linewidth=1.5, alpha=0.7)
ax3.axvline(n*p, color=MLGREEN, linestyle='--', linewidth=2, label=f'Expected = {n*p:.0f}')

ax3.set_title('P(At Least k Profitable Days) - Trading Example', fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Number of Profitable Days (k)', fontsize=10)
ax3.set_ylabel('Probability', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Normal approximation
ax4 = axes[1, 1]
n, p = 50, 0.4
x = np.arange(0, n+1)
pmf = stats.binom.pmf(x, n, p)

ax4.bar(x, pmf, color=MLBLUE, alpha=0.5, edgecolor='black', label='Binomial')

# Normal approximation
mu = n * p
sigma = np.sqrt(n * p * (1-p))
x_cont = np.linspace(0, n, 200)
normal_approx = stats.norm.pdf(x_cont, mu, sigma)

ax4.plot(x_cont, normal_approx, color=MLRED, linewidth=2.5, label=f'Normal(mu={mu:.1f}, sigma={sigma:.1f})')

ax4.set_title('Normal Approximation (n*p > 10)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Number of Successes', fontsize=10)
ax4.set_ylabel('Probability', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Add rule of thumb
ax4.text(0.95, 0.95, 'Rule: Use Normal if np > 10\nand n(1-p) > 10',
         transform=ax4.transAxes, fontsize=9, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
