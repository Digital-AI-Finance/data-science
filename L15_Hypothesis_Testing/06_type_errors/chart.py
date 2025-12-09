"""Type I and Type II Errors - False positives and negatives"""
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
fig.suptitle('Type I and Type II Errors', fontsize=14, fontweight='bold', color=MLPURPLE)

x = np.linspace(-4, 8, 300)

# Plot 1: Type I error (false positive)
ax1 = axes[0, 0]
y_h0 = stats.norm.pdf(x, loc=0)

ax1.plot(x, y_h0, color=MLBLUE, linewidth=2.5, label='H0 distribution')
ax1.fill_between(x, y_h0, alpha=0.3, color=MLBLUE)

# Critical region
alpha = 0.05
z_crit = stats.norm.ppf(1 - alpha)
ax1.fill_between(x, y_h0, where=x >= z_crit, alpha=0.7, color=MLRED, label=f'Type I Error (alpha = {alpha})')
ax1.axvline(z_crit, color=MLRED, linewidth=2, linestyle='--')

ax1.set_title('Type I Error: Rejecting True H0', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Test Statistic', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.annotate('False Positive\nalpha = P(reject H0 | H0 true)', xy=(2.5, 0.05),
             fontsize=9, color=MLRED, ha='center')

# Plot 2: Type II error (false negative)
ax2 = axes[0, 1]
# True effect exists
mu1 = 2  # True mean under H1
y_h0 = stats.norm.pdf(x, loc=0)
y_h1 = stats.norm.pdf(x, loc=mu1)

ax2.plot(x, y_h0, color=MLBLUE, linewidth=2.5, label='H0: mu = 0')
ax2.plot(x, y_h1, color=MLGREEN, linewidth=2.5, label=f'H1: mu = {mu1}')

z_crit = stats.norm.ppf(1 - 0.05)
ax2.axvline(z_crit, color=MLRED, linewidth=2, linestyle='--', label='Critical value')

# Beta is area of H1 to the left of critical value
beta = stats.norm.cdf(z_crit, loc=mu1)
ax2.fill_between(x, y_h1, where=x <= z_crit, alpha=0.5, color=MLORANGE, label=f'Type II Error (beta = {beta:.2f})')

ax2.set_title('Type II Error: Failing to Reject False H0', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Test Statistic', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Confusion matrix style
ax3 = axes[1, 0]
ax3.axis('off')

ax3.text(0.5, 0.95, 'Decision Matrix', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax3.transAxes)

# Create table
table_data = [
    ['', 'H0 True', 'H0 False'],
    ['Reject H0', 'Type I Error\n(False Positive)\nalpha', 'Correct!\n(True Positive)\n1 - beta (Power)'],
    ['Fail to Reject', 'Correct!\n(True Negative)\n1 - alpha', 'Type II Error\n(False Negative)\nbeta'],
]

colors = [
    ['white', MLLAVENDER, MLLAVENDER],
    [MLLAVENDER, '#FFB3B3', '#B3FFB3'],
    [MLLAVENDER, '#B3FFB3', '#FFFFB3'],
]

table = ax3.table(cellText=table_data, cellColours=colors, loc='center',
                  cellLoc='center', bbox=[0.1, 0.2, 0.8, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Plot 4: Power and sample size
ax4 = axes[1, 1]

# Power as function of sample size
sample_sizes = np.arange(10, 200, 5)
effect_size = 0.5  # Cohen's d

powers = []
for n in sample_sizes:
    se = 1 / np.sqrt(n)  # Standard error
    z_crit = stats.norm.ppf(0.95)  # One-tailed alpha = 0.05
    z_beta = (effect_size - z_crit * se) / se
    power = stats.norm.cdf(z_beta)
    powers.append(power)

ax4.plot(sample_sizes, powers, color=MLBLUE, linewidth=2.5)
ax4.axhline(0.8, color=MLGREEN, linewidth=2, linestyle='--', label='Power = 0.80 (conventional)')
ax4.fill_between(sample_sizes, powers, 0.8, where=np.array(powers) >= 0.8, alpha=0.3, color=MLGREEN)

# Find sample size for 80% power
idx_80 = np.argmin(np.abs(np.array(powers) - 0.8))
n_80 = sample_sizes[idx_80]
ax4.axvline(n_80, color=MLORANGE, linewidth=2, linestyle=':', label=f'n = {n_80} for 80% power')

ax4.set_title('Statistical Power vs Sample Size', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Sample Size (n)', fontsize=10)
ax4.set_ylabel('Power (1 - beta)', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
