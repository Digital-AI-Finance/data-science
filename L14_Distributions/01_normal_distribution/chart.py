"""Normal Distribution - The Gaussian Bell Curve"""
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
fig.suptitle('The Normal (Gaussian) Distribution', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Standard normal distribution
ax1 = axes[0, 0]
x = np.linspace(-4, 4, 200)
y = stats.norm.pdf(x)

ax1.plot(x, y, color=MLBLUE, linewidth=2.5)
ax1.fill_between(x, y, alpha=0.3, color=MLBLUE)

# Mark standard deviations
for i, color in zip([1, 2, 3], [MLGREEN, MLORANGE, MLRED]):
    ax1.axvline(i, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(-i, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

ax1.annotate('68%', xy=(0, 0.15), fontsize=10, ha='center', color=MLGREEN, fontweight='bold')
ax1.annotate('95%', xy=(0, 0.08), fontsize=10, ha='center', color=MLORANGE, fontweight='bold')
ax1.annotate('99.7%', xy=(0, 0.02), fontsize=10, ha='center', color=MLRED, fontweight='bold')

ax1.set_title('Standard Normal: N(0, 1)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Standard Deviations (z-score)', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Different means and standard deviations
ax2 = axes[0, 1]
params = [(0, 1, MLBLUE, 'N(0, 1)'),
          (2, 1, MLGREEN, 'N(2, 1)'),
          (0, 2, MLORANGE, 'N(0, 4)'),
          (-1, 0.5, MLRED, 'N(-1, 0.25)')]

x = np.linspace(-6, 6, 200)
for mu, sigma, color, label in params:
    y = stats.norm.pdf(x, mu, sigma)
    ax2.plot(x, y, color=color, linewidth=2, label=label)

ax2.set_title('Effect of Mean and Variance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Probability Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: 68-95-99.7 Rule visualization
ax3 = axes[1, 0]
x = np.linspace(-4, 4, 200)
y = stats.norm.pdf(x)

ax3.plot(x, y, color='black', linewidth=2)

# Fill regions
ax3.fill_between(x, y, where=(x >= -3) & (x <= 3), alpha=0.2, color=MLRED, label='99.7%')
ax3.fill_between(x, y, where=(x >= -2) & (x <= 2), alpha=0.3, color=MLORANGE, label='95%')
ax3.fill_between(x, y, where=(x >= -1) & (x <= 1), alpha=0.4, color=MLGREEN, label='68%')

ax3.set_title('The Empirical Rule (68-95-99.7)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Standard Deviations', fontsize=10)
ax3.set_ylabel('Probability Density', fontsize=10)
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(alpha=0.3)

# Add percentages
ax3.annotate('68%', xy=(0, 0.2), fontsize=12, ha='center', fontweight='bold')
ax3.annotate('95%', xy=(0, 0.05), fontsize=11, ha='center')
ax3.annotate('99.7%', xy=(0, 0.01), fontsize=10, ha='center')

# Plot 4: Sampling from normal
ax4 = axes[1, 1]
sample_sizes = [10, 100, 1000, 10000]
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]

for n, color in zip(sample_sizes, colors):
    sample = np.random.normal(100, 15, n)
    ax4.hist(sample, bins=30, density=True, alpha=0.4, color=color, label=f'n={n}', edgecolor='black')

# True distribution
x = np.linspace(40, 160, 200)
y = stats.norm.pdf(x, 100, 15)
ax4.plot(x, y, color=MLPURPLE, linewidth=3, linestyle='--', label='True N(100, 225)')

ax4.set_title('Sampling Convergence', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Value', fontsize=10)
ax4.set_ylabel('Density', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
