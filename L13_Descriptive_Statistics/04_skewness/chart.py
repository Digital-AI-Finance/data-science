"""Skewness - Asymmetry of distributions"""
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
fig.suptitle('Skewness: Measuring Asymmetry', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Three types of skewness
ax1 = axes[0, 0]
x = np.linspace(-4, 8, 200)

# Left-skewed (negatively skewed)
left_skewed = stats.skewnorm.pdf(x, -5, loc=4, scale=1.5)
ax1.plot(x, left_skewed, color=MLGREEN, linewidth=2.5, label='Left-Skewed (Negative)')
ax1.fill_between(x, left_skewed, alpha=0.2, color=MLGREEN)

# Symmetric
symmetric = stats.norm.pdf(x, loc=2, scale=1)
ax1.plot(x, symmetric, color=MLBLUE, linewidth=2.5, label='Symmetric (Skew = 0)')
ax1.fill_between(x, symmetric, alpha=0.2, color=MLBLUE)

# Right-skewed (positively skewed)
right_skewed = stats.skewnorm.pdf(x, 5, loc=0, scale=1.5)
ax1.plot(x, right_skewed, color=MLRED, linewidth=2.5, label='Right-Skewed (Positive)')
ax1.fill_between(x, right_skewed, alpha=0.2, color=MLRED)

ax1.set_title('Types of Skewness', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Stock returns skewness
ax2 = axes[0, 1]
# Simulate typical stock returns (slightly negative skew with fat tails)
returns_normal = np.random.normal(0.05, 2, 1000)
returns_crash = np.concatenate([np.random.normal(0.05, 1.5, 950), np.random.normal(-8, 2, 50)])

ax2.hist(returns_normal, bins=40, density=True, alpha=0.5, color=MLBLUE,
         edgecolor='black', label=f'Normal (Skew={stats.skew(returns_normal):.2f})')
ax2.hist(returns_crash, bins=40, density=True, alpha=0.5, color=MLRED,
         edgecolor='black', label=f'Crash Risk (Skew={stats.skew(returns_crash):.2f})')

ax2.axvline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title('Stock Returns: Negative Skew = Crash Risk', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Return (%)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.annotate('Left tail\n(crashes)', xy=(-10, 0.1), fontsize=9, color=MLRED, ha='center')

# Plot 3: Skewness by sector
ax3 = axes[1, 0]
sectors = ['Tech', 'Utilities', 'Finance', 'Healthcare', 'Energy']
skewness_values = [-0.3, 0.1, -0.8, -0.2, 0.5]
colors = [MLRED if s < 0 else MLGREEN for s in skewness_values]

bars = ax3.barh(sectors, skewness_values, color=colors, alpha=0.7, edgecolor='black')
ax3.axvline(0, color='black', linewidth=1)
ax3.set_title('Return Skewness by Sector', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Skewness', fontsize=10)

for bar, val in zip(bars, skewness_values):
    x_pos = val + 0.05 if val > 0 else val - 0.15
    ax3.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
             va='center', fontsize=9, fontweight='bold')

ax3.text(-0.5, -0.8, 'Negative = More crash risk', fontsize=9, color=MLRED, transform=ax3.transData)
ax3.text(0.3, -0.8, 'Positive = Upside potential', fontsize=9, color=MLGREEN, transform=ax3.transData)
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Interpretation guide
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Skewness Interpretation', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

interpretations = [
    ('Skew = 0', 'Symmetric distribution', 'Mean = Median', MLBLUE),
    ('Skew < 0', 'Left-skewed (Negative)', 'Long left tail, Mean < Median', MLRED),
    ('Skew > 0', 'Right-skewed (Positive)', 'Long right tail, Mean > Median', MLGREEN),
]

y = 0.75
for val, name, desc, color in interpretations:
    ax4.text(0.1, y, val, fontsize=12, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.3, y, name, fontsize=11, transform=ax4.transAxes)
    ax4.text(0.3, y - 0.06, desc, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.2

# Finance context
ax4.text(0.5, 0.15, 'Finance: Most stock returns have negative skewness', ha='center',
         fontsize=11, style='italic', color=MLPURPLE, transform=ax4.transAxes)
ax4.text(0.5, 0.05, '(Small gains common, rare large losses)', ha='center',
         fontsize=10, color='gray', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
