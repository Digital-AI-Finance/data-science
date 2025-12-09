"""Hypothesis Testing Concept - The basic framework"""
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
fig.suptitle('Hypothesis Testing: The Framework', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: The hypothesis testing process
ax1 = axes[0, 0]
ax1.axis('off')

steps = [
    ('1. State Hypotheses', 'H0: null hypothesis (status quo)\nH1: alternative (what we want to prove)'),
    ('2. Choose Significance', 'alpha = 0.05 (5% false positive rate)'),
    ('3. Collect Data', 'Sample from population'),
    ('4. Calculate Test Statistic', 'z, t, chi-squared, F, etc.'),
    ('5. Make Decision', 'Reject H0 if p-value < alpha'),
]

y = 0.9
for step, desc in steps:
    ax1.text(0.05, y, step, fontsize=11, fontweight='bold', color=MLPURPLE, transform=ax1.transAxes)
    ax1.text(0.05, y - 0.07, desc, fontsize=9, color='gray', transform=ax1.transAxes)
    y -= 0.18

ax1.set_title('Hypothesis Testing Process', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual representation
ax2 = axes[0, 1]
x = np.linspace(-4, 4, 200)
y_dist = stats.norm.pdf(x)

ax2.plot(x, y_dist, color=MLBLUE, linewidth=2.5)
ax2.fill_between(x, y_dist, alpha=0.3, color=MLBLUE)

# Critical regions (two-tailed, alpha=0.05)
alpha = 0.05
z_crit = stats.norm.ppf(1 - alpha/2)

ax2.fill_between(x, y_dist, where=x <= -z_crit, alpha=0.5, color=MLRED, label=f'Reject H0 (alpha/2={alpha/2})')
ax2.fill_between(x, y_dist, where=x >= z_crit, alpha=0.5, color=MLRED)
ax2.axvline(-z_crit, color=MLRED, linestyle='--', linewidth=2)
ax2.axvline(z_crit, color=MLRED, linestyle='--', linewidth=2)

# Sample test statistic
test_stat = 2.5
ax2.axvline(test_stat, color=MLGREEN, linewidth=3, label=f'Test statistic = {test_stat}')

ax2.set_title('Two-Tailed Test (alpha = 0.05)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Test Statistic (z-score)', fontsize=10)
ax2.set_ylabel('Probability Density', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.annotate(f'z_crit = {z_crit:.2f}', xy=(z_crit, 0.05), fontsize=9, ha='center')

# Plot 3: Finance example setup
ax3 = axes[1, 0]
np.random.seed(42)

# Generate sample returns
sample_returns = np.random.normal(0.08, 2.0, 50)  # Mean 0.08%, std 2%

ax3.hist(sample_returns, bins=15, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')
ax3.axvline(np.mean(sample_returns), color=MLGREEN, linewidth=2.5,
            label=f'Sample mean = {np.mean(sample_returns):.2f}%')
ax3.axvline(0, color=MLRED, linewidth=2.5, linestyle='--', label='H0: mu = 0')

ax3.set_title('Finance Example: Are Returns > 0?', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Daily Return (%)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Perform t-test
t_stat, p_value = stats.ttest_1samp(sample_returns, 0)
ax3.text(0.95, 0.95, f't-stat = {t_stat:.2f}\np-value = {p_value:.4f}',
         transform=ax3.transAxes, fontsize=10, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))

# Plot 4: Decision flowchart
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Decision Rule', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

# Decision boxes
from matplotlib.patches import FancyBboxPatch
box1 = FancyBboxPatch((0.15, 0.5), 0.3, 0.3, boxstyle="round,pad=0.05",
                       edgecolor=MLGREEN, facecolor='white', linewidth=2, transform=ax4.transAxes)
ax4.add_patch(box1)
ax4.text(0.3, 0.65, 'p-value < alpha', ha='center', fontsize=11, fontweight='bold',
         color=MLGREEN, transform=ax4.transAxes)
ax4.text(0.3, 0.55, 'REJECT H0\nEvidence supports H1', ha='center', fontsize=9,
         color='gray', transform=ax4.transAxes)

box2 = FancyBboxPatch((0.55, 0.5), 0.3, 0.3, boxstyle="round,pad=0.05",
                       edgecolor=MLRED, facecolor='white', linewidth=2, transform=ax4.transAxes)
ax4.add_patch(box2)
ax4.text(0.7, 0.65, 'p-value >= alpha', ha='center', fontsize=11, fontweight='bold',
         color=MLRED, transform=ax4.transAxes)
ax4.text(0.7, 0.55, 'FAIL TO REJECT H0\nInsufficient evidence', ha='center', fontsize=9,
         color='gray', transform=ax4.transAxes)

ax4.text(0.5, 0.25, 'Common alpha values: 0.05 (5%), 0.01 (1%), 0.10 (10%)',
         ha='center', fontsize=10, transform=ax4.transAxes)
ax4.text(0.5, 0.1, 'Note: "Fail to reject" is NOT the same as "accept H0"',
         ha='center', fontsize=10, style='italic', color=MLRED, transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
