"""QQ Plot - Quantile-Quantile Plot for Distribution Checking"""
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
fig.suptitle('QQ Plots: Checking Distribution Fit', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Normal data - good fit
ax1 = axes[0, 0]
normal_data = np.random.normal(0, 1, 500)
stats.probplot(normal_data, dist="norm", plot=ax1)
ax1.get_lines()[0].set_color(MLBLUE)
ax1.get_lines()[0].set_markersize(4)
ax1.get_lines()[1].set_color(MLRED)
ax1.get_lines()[1].set_linewidth(2)

ax1.set_title('Normal Data: Points Follow Line', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Theoretical Quantiles', fontsize=10)
ax1.set_ylabel('Sample Quantiles', fontsize=10)
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, 'GOOD FIT: Normal distribution', transform=ax1.transAxes,
         fontsize=9, va='top', color=MLGREEN, fontweight='bold')

# Plot 2: Fat tails - stock returns
ax2 = axes[0, 1]
fat_tail_data = np.random.standard_t(3, 500) * 2  # t-distribution (fat tails)
stats.probplot(fat_tail_data, dist="norm", plot=ax2)
ax2.get_lines()[0].set_color(MLBLUE)
ax2.get_lines()[0].set_markersize(4)
ax2.get_lines()[1].set_color(MLRED)
ax2.get_lines()[1].set_linewidth(2)

ax2.set_title('Fat Tails: S-curve at Extremes', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Theoretical Quantiles', fontsize=10)
ax2.set_ylabel('Sample Quantiles', fontsize=10)
ax2.grid(alpha=0.3)

# Annotate the deviations
ax2.annotate('Heavy left tail', xy=(-2.5, -6), fontsize=9, color=MLRED)
ax2.annotate('Heavy right tail', xy=(1.5, 6), fontsize=9, color=MLRED)
ax2.text(0.05, 0.95, 'FAT TAILS: More extreme values\nthan Normal predicts', transform=ax2.transAxes,
         fontsize=9, va='top', color=MLRED, fontweight='bold')

# Plot 3: Right skewed data
ax3 = axes[1, 0]
skewed_data = np.random.exponential(2, 500)
stats.probplot(skewed_data, dist="norm", plot=ax3)
ax3.get_lines()[0].set_color(MLBLUE)
ax3.get_lines()[0].set_markersize(4)
ax3.get_lines()[1].set_color(MLRED)
ax3.get_lines()[1].set_linewidth(2)

ax3.set_title('Right Skewed: Curve Up', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Theoretical Quantiles', fontsize=10)
ax3.set_ylabel('Sample Quantiles', fontsize=10)
ax3.grid(alpha=0.3)
ax3.text(0.05, 0.95, 'RIGHT SKEW: Long right tail', transform=ax3.transAxes,
         fontsize=9, va='top', color=MLORANGE, fontweight='bold')

# Plot 4: Interpretation guide
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'QQ Plot Interpretation Guide', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

patterns = [
    ('Points on line', 'Data follows normal distribution', MLGREEN),
    ('S-curve (both ends deviate)', 'Fat tails (leptokurtic) - common in finance!', MLRED),
    ('Curve up (right end above)', 'Right skewed (positive skew)', MLORANGE),
    ('Curve down (right end below)', 'Left skewed (negative skew)', MLBLUE),
    ('Points above line', 'Heavier tails than normal', MLRED),
    ('Points below line', 'Lighter tails than normal', MLGREEN),
]

y = 0.8
for pattern, meaning, color in patterns:
    ax4.text(0.05, y, pattern + ':', fontsize=10, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.05, y - 0.05, meaning, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.12

ax4.text(0.5, 0.08, 'Use: stats.probplot(data, dist="norm", plot=ax)',
         ha='center', fontsize=10, family='monospace', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
