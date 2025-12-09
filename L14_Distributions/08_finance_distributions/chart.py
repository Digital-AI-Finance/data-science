"""Finance Distributions - Common distributions in finance"""
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
fig.suptitle('Distributions in Finance', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Log-normal for prices
ax1 = axes[0, 0]
x = np.linspace(0.01, 5, 200)

# Log-normal parameters
for sigma, color, label in [(0.25, MLBLUE, 'Low Vol (sigma=0.25)'),
                            (0.5, MLORANGE, 'Med Vol (sigma=0.5)'),
                            (1.0, MLRED, 'High Vol (sigma=1.0)')]:
    y = stats.lognorm.pdf(x, sigma)
    ax1.plot(x, y, color=color, linewidth=2.5, label=label)

ax1.set_title('Log-Normal: Stock Prices', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Price (relative to initial)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.text(0.95, 0.95, 'Prices are positive\nand right-skewed', transform=ax1.transAxes,
         fontsize=9, va='top', ha='right', style='italic')

# Plot 2: Chi-squared for volatility
ax2 = axes[0, 1]
x = np.linspace(0, 20, 200)

for df, color, label in [(2, MLBLUE, 'df=2'),
                          (4, MLORANGE, 'df=4'),
                          (8, MLGREEN, 'df=8')]:
    y = stats.chi2.pdf(x, df)
    ax2.plot(x, y, color=color, linewidth=2.5, label=label)

ax2.set_title('Chi-squared: Variance Ratios', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.text(0.95, 0.95, 'Used in volatility\nand variance tests', transform=ax2.transAxes,
         fontsize=9, va='top', ha='right', style='italic')

# Plot 3: Poisson for event counts
ax3 = axes[1, 0]
x = np.arange(0, 15)

for lam, color, label in [(1, MLBLUE, 'lambda=1 (rare)'),
                          (3, MLORANGE, 'lambda=3'),
                          (7, MLGREEN, 'lambda=7 (frequent)')]:
    pmf = stats.poisson.pmf(x, lam)
    ax3.bar(x + 0.25 * (lam - 3) / 3, pmf, width=0.25, color=color, alpha=0.7, label=label, edgecolor='black')

ax3.set_title('Poisson: Event Counts (Defaults, Trades)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Number of Events', fontsize=10)
ax3.set_ylabel('Probability', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Distribution Applications in Finance', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

applications = [
    ('Normal', 'Returns (short-term)', 'Mean-variance optimization', MLBLUE),
    ('Log-Normal', 'Stock prices', 'Option pricing (Black-Scholes)', MLGREEN),
    ('t-distribution', 'Returns (fat tails)', 'Risk management, VaR', MLRED),
    ('Chi-squared', 'Variance ratios', 'Volatility tests', MLORANGE),
    ('Poisson', 'Event counts', 'Default modeling, trade arrivals', MLPURPLE),
    ('Exponential', 'Time between events', 'Duration models', MLBLUE),
]

y = 0.8
for dist, use_case, application, color in applications:
    ax4.text(0.05, y, dist + ':', fontsize=10, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.25, y, use_case, fontsize=9, transform=ax4.transAxes)
    ax4.text(0.55, y, application, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.11

ax4.text(0.5, 0.08, 'Key: Choose distribution based on data characteristics!',
         ha='center', fontsize=10, fontweight='bold', color=MLPURPLE, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
