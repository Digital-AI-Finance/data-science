"""T-Test - Testing means"""
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
fig.suptitle('T-Tests: Comparing Means', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: One-sample t-test
ax1 = axes[0, 0]
sample = np.random.normal(0.15, 2.0, 30)  # Strategy returns

ax1.hist(sample, bins=12, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')
ax1.axvline(np.mean(sample), color=MLGREEN, linewidth=2.5, label=f'Sample mean = {np.mean(sample):.2f}')
ax1.axvline(0, color=MLRED, linewidth=2.5, linestyle='--', label='H0: mu = 0')

t_stat, p_val = stats.ttest_1samp(sample, 0)
ax1.set_title('One-Sample T-Test: Is mean != 0?', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Return (%)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.text(0.95, 0.95, f't = {t_stat:.2f}\np = {p_val:.4f}', transform=ax1.transAxes,
         fontsize=10, va='top', ha='right', bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))

# Plot 2: Two-sample t-test (independent)
ax2 = axes[0, 1]
strategy_a = np.random.normal(0.10, 1.5, 50)
strategy_b = np.random.normal(0.25, 1.8, 50)

bins = np.linspace(-5, 5, 20)
ax2.hist(strategy_a, bins=bins, density=True, alpha=0.5, color=MLBLUE, edgecolor='black', label='Strategy A')
ax2.hist(strategy_b, bins=bins, density=True, alpha=0.5, color=MLGREEN, edgecolor='black', label='Strategy B')
ax2.axvline(np.mean(strategy_a), color=MLBLUE, linewidth=2, linestyle='--')
ax2.axvline(np.mean(strategy_b), color=MLGREEN, linewidth=2, linestyle='--')

t_stat, p_val = stats.ttest_ind(strategy_a, strategy_b)
ax2.set_title('Two-Sample T-Test: Are means different?', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Return (%)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.text(0.95, 0.95, f't = {t_stat:.2f}\np = {p_val:.4f}', transform=ax2.transAxes,
         fontsize=10, va='top', ha='right', bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))

# Plot 3: Paired t-test
ax3 = axes[1, 0]
before = np.random.normal(100, 10, 20)  # Portfolio value before
after = before + np.random.normal(5, 8, 20)  # After optimization

x_pos = np.arange(len(before))
ax3.scatter(x_pos, before, color=MLBLUE, s=60, label='Before', zorder=5)
ax3.scatter(x_pos, after, color=MLGREEN, s=60, label='After', zorder=5)

for i in range(len(before)):
    color = MLGREEN if after[i] > before[i] else MLRED
    ax3.plot([i, i], [before[i], after[i]], color=color, linewidth=1.5, alpha=0.7)

t_stat, p_val = stats.ttest_rel(before, after)
ax3.set_title('Paired T-Test: Before vs After', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Client ID', fontsize=10)
ax3.set_ylabel('Portfolio Value ($)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.text(0.95, 0.05, f't = {t_stat:.2f}\np = {p_val:.4f}', transform=ax3.transAxes,
         fontsize=10, va='bottom', ha='right', bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))

# Plot 4: T-test summary
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'T-Test Summary', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

tests = [
    ('One-Sample', 'stats.ttest_1samp(sample, mu0)', 'Compare sample mean to known value', MLBLUE),
    ('Two-Sample', 'stats.ttest_ind(sample1, sample2)', 'Compare means of two groups', MLGREEN),
    ('Paired', 'stats.ttest_rel(before, after)', 'Compare paired observations', MLORANGE),
    ('Welch', 'stats.ttest_ind(..., equal_var=False)', 'Unequal variances', MLRED),
]

y = 0.78
for name, code, desc, color in tests:
    ax4.text(0.05, y, name + ':', fontsize=11, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.25, y, code, fontsize=9, family='monospace', transform=ax4.transAxes)
    ax4.text(0.25, y - 0.05, desc, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.17

ax4.text(0.5, 0.12, 'Assumptions: Normal distribution (or large n > 30)',
         ha='center', fontsize=10, transform=ax4.transAxes)
ax4.text(0.5, 0.02, 'For non-normal: use Mann-Whitney U or Wilcoxon tests',
         ha='center', fontsize=9, style='italic', color='gray', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
