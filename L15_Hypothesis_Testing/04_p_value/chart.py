"""P-Value - Probability of observing result under H0"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Understanding P-Values', fontsize=14, fontweight='bold', color=MLPURPLE)

x = np.linspace(-4, 4, 200)
y = stats.norm.pdf(x)

# Plot 1: P-value visualization (two-tailed)
ax1 = axes[0, 0]
test_stat = 2.1
p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))

ax1.plot(x, y, color=MLBLUE, linewidth=2.5)
ax1.fill_between(x, y, where=x <= -abs(test_stat), alpha=0.5, color=MLRED)
ax1.fill_between(x, y, where=x >= abs(test_stat), alpha=0.5, color=MLRED)
ax1.axvline(test_stat, color=MLGREEN, linewidth=2.5, label=f'Test stat = {test_stat}')
ax1.axvline(-test_stat, color=MLGREEN, linewidth=2.5, linestyle='--')

ax1.set_title(f'P-value = {p_value:.4f} (Two-tailed)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Test Statistic (z)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.text(2.5, 0.15, f'Area = {p_value/2:.4f}', fontsize=9, color=MLRED)
ax1.text(-3.2, 0.15, f'Area = {p_value/2:.4f}', fontsize=9, color=MLRED)

# Plot 2: Different test statistics
ax2 = axes[0, 1]
test_stats = [1.0, 1.96, 2.5, 3.0]
p_values = [2 * (1 - stats.norm.cdf(abs(t))) for t in test_stats]
colors = [MLGREEN if p >= 0.05 else MLRED for p in p_values]

ax2.barh(range(len(test_stats)), p_values, color=colors, alpha=0.7, edgecolor='black')
ax2.axvline(0.05, color=MLPURPLE, linewidth=2.5, linestyle='--', label='alpha = 0.05')
ax2.axvline(0.01, color=MLORANGE, linewidth=2, linestyle=':', label='alpha = 0.01')

for i, (t, p) in enumerate(zip(test_stats, p_values)):
    ax2.text(p + 0.01, i, f'p = {p:.4f}', va='center', fontsize=9)

ax2.set_yticks(range(len(test_stats)))
ax2.set_yticklabels([f'z = {t}' for t in test_stats])
ax2.set_title('P-values for Different Test Statistics', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('P-value', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: P-value interpretation
ax3 = axes[1, 0]
ax3.axis('off')

ax3.text(0.5, 0.95, 'P-Value Interpretation', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax3.transAxes)

interpretations = [
    ('p > 0.10', 'No evidence against H0', MLGREEN),
    ('0.05 < p < 0.10', 'Weak evidence against H0', MLORANGE),
    ('0.01 < p < 0.05', 'Moderate evidence against H0', MLORANGE),
    ('0.001 < p < 0.01', 'Strong evidence against H0', MLRED),
    ('p < 0.001', 'Very strong evidence against H0', MLRED),
]

y = 0.78
for range_str, interp, color in interpretations:
    ax3.text(0.1, y, range_str, fontsize=11, fontweight='bold', color=color, transform=ax3.transAxes)
    ax3.text(0.4, y, interp, fontsize=10, transform=ax3.transAxes)
    y -= 0.14

ax3.text(0.5, 0.1, 'P-value = probability of seeing this result (or more extreme)\nif H0 is true',
         ha='center', fontsize=10, style='italic', color='gray', transform=ax3.transAxes)

# Plot 4: Common mistakes
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'P-Value: What It Is NOT', ha='center', fontsize=14,
         fontweight='bold', color=MLRED, transform=ax4.transAxes)

mistakes = [
    'Probability that H0 is true',
    'Probability that H1 is true',
    'Probability of making an error',
    'Size of the effect',
    'Importance of the finding',
]

y = 0.78
for mistake in mistakes:
    ax4.text(0.1, y, 'X', fontsize=14, fontweight='bold', color=MLRED, transform=ax4.transAxes)
    ax4.text(0.2, y, mistake, fontsize=10, transform=ax4.transAxes)
    y -= 0.12

ax4.text(0.5, 0.2, 'P-value IS: Probability of data (or more extreme)\ngiven H0 is true: P(Data | H0)',
         ha='center', fontsize=11, color=MLGREEN, fontweight='bold', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLGREEN))

ax4.text(0.5, 0.02, 'Statistical significance != Practical significance',
         ha='center', fontsize=10, style='italic', color=MLPURPLE, transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
