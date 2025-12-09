"""Fat Tails in Finance - Why Normal Fails"""
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
fig.suptitle('Fat Tails in Finance: Why Normal Distributions Fail', fontsize=14,
             fontweight='bold', color=MLPURPLE)

x = np.linspace(-6, 6, 300)

# Plot 1: Normal vs t-distribution
ax1 = axes[0, 0]
normal = stats.norm.pdf(x)
t_3 = stats.t.pdf(x, df=3)
t_5 = stats.t.pdf(x, df=5)

ax1.plot(x, normal, color=MLBLUE, linewidth=2.5, label='Normal')
ax1.plot(x, t_3, color=MLRED, linewidth=2.5, label='t (df=3)')
ax1.plot(x, t_5, color=MLORANGE, linewidth=2.5, label='t (df=5)')

ax1.set_title('Normal vs Student-t Distributions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Log scale tail comparison
ax2 = axes[0, 1]
ax2.semilogy(x, normal, color=MLBLUE, linewidth=2.5, label='Normal')
ax2.semilogy(x, t_3, color=MLRED, linewidth=2.5, label='t (df=3)')

# Highlight the difference
ax2.fill_between(x, t_3, normal, where=np.abs(x) > 2, alpha=0.3, color=MLRED)

ax2.set_title('Tail Probability (Log Scale)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Log Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.annotate('10-100x more\nlikely with\nfat tails!', xy=(4, 0.01), fontsize=9, color=MLRED,
             ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLRED))

# Plot 3: Probability of extreme events
ax3 = axes[1, 0]
sigmas = [2, 3, 4, 5, 6]
normal_probs = [2 * (1 - stats.norm.cdf(s)) for s in sigmas]
t3_probs = [2 * (1 - stats.t.cdf(s, df=3)) for s in sigmas]
t5_probs = [2 * (1 - stats.t.cdf(s, df=5)) for s in sigmas]

x_pos = np.arange(len(sigmas))
width = 0.25

ax3.bar(x_pos - width, normal_probs, width, color=MLBLUE, alpha=0.7, label='Normal', edgecolor='black')
ax3.bar(x_pos, t5_probs, width, color=MLORANGE, alpha=0.7, label='t (df=5)', edgecolor='black')
ax3.bar(x_pos + width, t3_probs, width, color=MLRED, alpha=0.7, label='t (df=3)', edgecolor='black')

ax3.set_yscale('log')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'{s}σ' for s in sigmas])
ax3.set_title('P(|X| > kσ) - Extreme Event Probability', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Threshold (Standard Deviations)', fontsize=10)
ax3.set_ylabel('Probability (log scale)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Add text for 6-sigma event
ax3.text(4, normal_probs[-1] * 10, f'6σ: Normal = 1 in 500M\nt(3) = 1 in 50',
         fontsize=8, color=MLRED, ha='center')

# Plot 4: Real-world implications
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Real-World Implications', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

implications = [
    ('Black Monday (1987)', '-22.6% in one day', '25+ sigma event under Normal\n(Should occur once per billion years!)', MLRED),
    ('2008 Crisis', 'Multiple 5+ sigma days', 'Normal predicts: once per 14,000 years', MLRED),
    ('Flash Crash (2010)', '-9% in minutes', 'Risk models completely failed', MLORANGE),
]

y = 0.78
for event, move, explanation, color in implications:
    ax4.text(0.05, y, event + ':', fontsize=11, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.35, y, move, fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, y - 0.08, explanation, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.22

ax4.text(0.5, 0.08, 'Key Lesson: Never trust Normal for risk management!',
         ha='center', fontsize=11, fontweight='bold', color=MLRED, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLRED))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
