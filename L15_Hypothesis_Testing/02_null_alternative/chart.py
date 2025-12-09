"""Null vs Alternative Hypothesis - Setting up the test"""
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
fig.suptitle('Null and Alternative Hypotheses', fontsize=14, fontweight='bold', color=MLPURPLE)

x = np.linspace(-4, 6, 200)

# Plot 1: Two-tailed test
ax1 = axes[0, 0]
y = stats.norm.pdf(x, loc=0, scale=1)
ax1.plot(x, y, color=MLBLUE, linewidth=2.5)
ax1.fill_between(x, y, alpha=0.3, color=MLBLUE)

# Critical regions
z_crit = 1.96
ax1.fill_between(x, y, where=x <= -z_crit, alpha=0.5, color=MLRED)
ax1.fill_between(x, y, where=x >= z_crit, alpha=0.5, color=MLRED)
ax1.axvline(0, color=MLPURPLE, linewidth=2, linestyle='--')

ax1.set_title('Two-Tailed: H1: mu != 0', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Test Statistic', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.grid(alpha=0.3)
ax1.text(0.5, 0.95, 'H0: mu = 0\nH1: mu != 0', transform=ax1.transAxes, fontsize=10, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLPURPLE))
ax1.annotate('Reject', xy=(-3, 0.02), fontsize=9, color=MLRED, ha='center')
ax1.annotate('Reject', xy=(3, 0.02), fontsize=9, color=MLRED, ha='center')

# Plot 2: Right-tailed test
ax2 = axes[0, 1]
ax2.plot(x, y, color=MLBLUE, linewidth=2.5)
ax2.fill_between(x, y, alpha=0.3, color=MLBLUE)

z_crit_right = stats.norm.ppf(0.95)
ax2.fill_between(x, y, where=x >= z_crit_right, alpha=0.5, color=MLRED)
ax2.axvline(0, color=MLPURPLE, linewidth=2, linestyle='--')

ax2.set_title('Right-Tailed: H1: mu > 0', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Test Statistic', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.grid(alpha=0.3)
ax2.text(0.5, 0.95, 'H0: mu <= 0\nH1: mu > 0', transform=ax2.transAxes, fontsize=10, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLPURPLE))
ax2.annotate('Reject', xy=(2.5, 0.02), fontsize=9, color=MLRED, ha='center')

# Plot 3: Left-tailed test
ax3 = axes[1, 0]
ax3.plot(x, y, color=MLBLUE, linewidth=2.5)
ax3.fill_between(x, y, alpha=0.3, color=MLBLUE)

z_crit_left = stats.norm.ppf(0.05)
ax3.fill_between(x, y, where=x <= z_crit_left, alpha=0.5, color=MLRED)
ax3.axvline(0, color=MLPURPLE, linewidth=2, linestyle='--')

ax3.set_title('Left-Tailed: H1: mu < 0', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Test Statistic', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.grid(alpha=0.3)
ax3.text(0.5, 0.95, 'H0: mu >= 0\nH1: mu < 0', transform=ax3.transAxes, fontsize=10, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLPURPLE))
ax3.annotate('Reject', xy=(-2.5, 0.02), fontsize=9, color=MLRED, ha='center')

# Plot 4: Finance examples
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Finance Hypothesis Examples', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

examples = [
    ('Strategy alpha', 'H0: alpha = 0 (no excess return)\nH1: alpha > 0 (positive alpha)', 'Right-tailed', MLGREEN),
    ('Market efficiency', 'H0: returns are random\nH1: returns are predictable', 'Two-tailed', MLBLUE),
    ('Risk reduction', 'H0: volatility unchanged\nH1: volatility decreased', 'Left-tailed', MLORANGE),
    ('Correlation change', 'H0: rho1 = rho2\nH1: rho1 != rho2', 'Two-tailed', MLRED),
]

y = 0.8
for title, hypothesis, test_type, color in examples:
    ax4.text(0.05, y, title + ':', fontsize=11, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.35, y, hypothesis, fontsize=9, transform=ax4.transAxes)
    ax4.text(0.75, y, test_type, fontsize=9, color='gray', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.3))
    y -= 0.18

ax4.text(0.5, 0.08, 'Choose test direction based on what you want to prove!',
         ha='center', fontsize=10, style='italic', color=MLPURPLE, transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
