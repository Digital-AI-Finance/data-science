"""Kurtosis - Tail heaviness of distributions"""
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
fig.suptitle('Kurtosis: Measuring Tail Heaviness', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Three types of kurtosis
ax1 = axes[0, 0]
x = np.linspace(-5, 5, 200)

# Platykurtic (light tails)
platy = stats.uniform.pdf(x, loc=-2, scale=4)
ax1.plot(x, platy, color=MLGREEN, linewidth=2.5, label='Platykurtic (Thin tails)')

# Mesokurtic (normal)
meso = stats.norm.pdf(x, loc=0, scale=1)
ax1.plot(x, meso, color=MLBLUE, linewidth=2.5, label='Mesokurtic (Normal)')

# Leptokurtic (heavy tails)
lepto = stats.t.pdf(x, df=3, loc=0, scale=0.7)
ax1.plot(x, lepto, color=MLRED, linewidth=2.5, label='Leptokurtic (Fat tails)')

ax1.set_title('Types of Kurtosis', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Highlight tails
ax1.annotate('Fat tails', xy=(3.5, 0.05), fontsize=9, color=MLRED, fontweight='bold')

# Plot 2: Focus on tails (log scale)
ax2 = axes[0, 1]
x = np.linspace(-6, 6, 200)

normal = stats.norm.pdf(x)
t_dist = stats.t.pdf(x, df=3)

ax2.semilogy(x, normal, color=MLBLUE, linewidth=2.5, label='Normal (Kurt=3)')
ax2.semilogy(x, t_dist, color=MLRED, linewidth=2.5, label='t-dist df=3 (Fat tails)')

ax2.set_title('Tail Comparison (Log Scale)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Log Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Highlight the difference
ax2.fill_between(x, t_dist, normal, where=np.abs(x) > 2.5, alpha=0.3, color=MLRED)
ax2.annotate('Extreme events\nmore likely!', xy=(4, 0.01), fontsize=9, color=MLRED, ha='center')

# Plot 3: Stock returns have excess kurtosis
ax3 = axes[1, 0]
# Simulate returns with fat tails
normal_returns = np.random.normal(0, 2, 1000)
fat_tail_returns = np.random.standard_t(4, 1000) * 1.5

ax3.hist(normal_returns, bins=50, density=True, alpha=0.5, color=MLBLUE,
         edgecolor='black', label=f'Normal (Kurt={stats.kurtosis(normal_returns)+3:.1f})')
ax3.hist(fat_tail_returns, bins=50, density=True, alpha=0.5, color=MLRED,
         edgecolor='black', label=f'Fat Tails (Kurt={stats.kurtosis(fat_tail_returns)+3:.1f})')

ax3.set_title('Stock Returns: Fat Tails Are Real', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Return (%)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Kurtosis interpretation
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Kurtosis Interpretation', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

# Excess kurtosis (relative to normal = 3)
interpretations = [
    ('Kurtosis = 3', 'Mesokurtic', 'Normal distribution (baseline)', MLBLUE),
    ('Kurtosis < 3', 'Platykurtic', 'Thin tails, fewer outliers', MLGREEN),
    ('Kurtosis > 3', 'Leptokurtic', 'Fat tails, more extreme events', MLRED),
]

y = 0.75
for val, name, desc, color in interpretations:
    ax4.text(0.1, y, val, fontsize=11, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.35, y, name, fontsize=11, transform=ax4.transAxes)
    ax4.text(0.35, y - 0.06, desc, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.2

# Finance warning
ax4.text(0.5, 0.15, 'WARNING: Stock returns typically have kurtosis > 10', ha='center',
         fontsize=11, fontweight='bold', color=MLRED, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLRED))
ax4.text(0.5, 0.02, 'Extreme moves happen more often than Normal assumes!', ha='center',
         fontsize=10, color='gray', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
