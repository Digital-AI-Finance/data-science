"""Confidence Intervals - Estimating parameters"""
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
fig.suptitle('Confidence Intervals', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: CI visualization
ax1 = axes[0, 0]
true_mean = 100
sample_size = 30
num_samples = 20

# Generate samples and CIs
np.random.seed(42)
contains_true = 0

for i in range(num_samples):
    sample = np.random.normal(true_mean, 15, sample_size)
    sample_mean = np.mean(sample)
    sem = stats.sem(sample)
    ci = stats.t.interval(0.95, sample_size-1, loc=sample_mean, scale=sem)

    color = MLGREEN if ci[0] <= true_mean <= ci[1] else MLRED
    if ci[0] <= true_mean <= ci[1]:
        contains_true += 1

    ax1.plot([ci[0], ci[1]], [i, i], color=color, linewidth=2)
    ax1.scatter([sample_mean], [i], color=color, s=40, zorder=5)

ax1.axvline(true_mean, color=MLPURPLE, linewidth=2.5, linestyle='--', label=f'True mean = {true_mean}')
ax1.set_title(f'95% CIs: {contains_true}/{num_samples} contain true mean', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Sample #', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Different confidence levels
ax2 = axes[0, 1]
sample = np.random.normal(100, 15, 50)
sample_mean = np.mean(sample)
sem = stats.sem(sample)

levels = [0.90, 0.95, 0.99]
colors = [MLGREEN, MLBLUE, MLRED]
y_positions = [3, 2, 1]

for level, color, y in zip(levels, colors, y_positions):
    ci = stats.t.interval(level, len(sample)-1, loc=sample_mean, scale=sem)
    ax2.fill_between([ci[0], ci[1]], [y-0.3, y-0.3], [y+0.3, y+0.3], color=color, alpha=0.5)
    ax2.plot([ci[0], ci[1]], [y, y], color=color, linewidth=3)
    ax2.scatter([sample_mean], [y], color='black', s=80, zorder=5)
    ax2.text(ci[1] + 0.5, y, f'{level*100:.0f}% CI: [{ci[0]:.1f}, {ci[1]:.1f}]',
             fontsize=9, va='center')

ax2.set_title('Wider CI = Higher Confidence', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Value', fontsize=10)
ax2.set_yticks([1, 2, 3])
ax2.set_yticklabels(['99%', '95%', '90%'])
ax2.grid(axis='x', alpha=0.3)

# Plot 3: CI for stock returns
ax3 = axes[1, 0]
returns = np.random.normal(0.08, 1.5, 252)  # Daily returns

mean_ret = np.mean(returns)
sem_ret = stats.sem(returns)
ci_95 = stats.t.interval(0.95, len(returns)-1, loc=mean_ret, scale=sem_ret)

ax3.hist(returns, bins=30, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')
ax3.axvline(mean_ret, color=MLGREEN, linewidth=2.5, label=f'Mean = {mean_ret:.3f}%')
ax3.axvline(ci_95[0], color=MLRED, linewidth=2, linestyle='--')
ax3.axvline(ci_95[1], color=MLRED, linewidth=2, linestyle='--')
ax3.axvspan(ci_95[0], ci_95[1], alpha=0.2, color=MLRED, label=f'95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]')

ax3.set_title('95% CI for Daily Return Mean', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Daily Return (%)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Check if 0 is in the CI
if ci_95[0] > 0:
    ax3.text(0.5, 0.95, 'CI > 0: Significant positive return!', transform=ax3.transAxes,
             fontsize=10, va='top', ha='center', color=MLGREEN, fontweight='bold')
elif ci_95[1] < 0:
    ax3.text(0.5, 0.95, 'CI < 0: Significant negative return!', transform=ax3.transAxes,
             fontsize=10, va='top', ha='center', color=MLRED, fontweight='bold')
else:
    ax3.text(0.5, 0.95, 'CI includes 0: Cannot reject H0', transform=ax3.transAxes,
             fontsize=10, va='top', ha='center', color=MLORANGE, fontweight='bold')

# Plot 4: CI formula and interpretation
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Confidence Interval Formula', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

ax4.text(0.5, 0.78, 'CI = sample_mean +/- t_critical * (std / sqrt(n))',
         ha='center', fontsize=12, family='monospace', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

interpretations = [
    ('95% CI means:', 'If we repeated sampling 100 times,\n~95 of the CIs would contain the true mean'),
    ('CI width depends on:', '1. Confidence level (higher = wider)\n2. Sample size (larger = narrower)\n3. Variability (higher = wider)'),
    ('CI vs p-value:', 'If 95% CI excludes H0 value,\nthen p-value < 0.05'),
]

y = 0.6
for title, desc in interpretations:
    ax4.text(0.1, y, title, fontsize=10, fontweight='bold', color=MLBLUE, transform=ax4.transAxes)
    ax4.text(0.1, y - 0.08, desc, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.22

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
