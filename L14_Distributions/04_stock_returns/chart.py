"""Stock Return Distributions - Real-world vs Normal"""
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
fig.suptitle('Stock Return Distributions', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate realistic stock returns (with fat tails)
n = 2520  # 10 years of daily data
returns_real = np.random.standard_t(4, n) * 1.5  # t-distribution (fat tails)
returns_normal = np.random.normal(np.mean(returns_real), np.std(returns_real), n)

# Plot 1: Histogram comparison
ax1 = axes[0, 0]
bins = np.linspace(-10, 10, 50)
ax1.hist(returns_real, bins=bins, density=True, alpha=0.5, color=MLBLUE,
         edgecolor='black', label='Real Returns')
ax1.hist(returns_normal, bins=bins, density=True, alpha=0.5, color=MLGREEN,
         edgecolor='black', label='Normal Returns')

# Overlay normal PDF
x = np.linspace(-10, 10, 200)
pdf = stats.norm.pdf(x, np.mean(returns_real), np.std(returns_real))
ax1.plot(x, pdf, color=MLRED, linewidth=2, linestyle='--', label='Normal PDF')

ax1.set_title('Real Returns Have Fatter Tails', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Daily Return (%)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Log scale to show tails
ax2 = axes[0, 1]
ax2.hist(returns_real, bins=bins, density=True, alpha=0.7, color=MLBLUE,
         edgecolor='black', label='Real Returns')
ax2.plot(x, pdf, color=MLRED, linewidth=2.5, label='Normal PDF')
ax2.set_yscale('log')

ax2.set_title('Log Scale: Fat Tails Visible', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Daily Return (%)', fontsize=10)
ax2.set_ylabel('Log Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Annotate the gap
ax2.annotate('Fat tails:\nExtreme events\nmore likely', xy=(6, 0.001), fontsize=9,
             color=MLRED, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLRED))

# Plot 3: Extreme events comparison
ax3 = axes[1, 0]
thresholds = np.arange(2, 7, 0.5)  # Standard deviations

# Count events beyond threshold
real_extreme = [np.sum(np.abs(returns_real) > t * np.std(returns_real)) / len(returns_real) * 100
                for t in thresholds]
normal_expected = [2 * (1 - stats.norm.cdf(t)) * 100 for t in thresholds]

ax3.bar(thresholds - 0.15, real_extreme, width=0.3, color=MLBLUE, alpha=0.7,
        label='Actual Returns', edgecolor='black')
ax3.bar(thresholds + 0.15, normal_expected, width=0.3, color=MLGREEN, alpha=0.7,
        label='Normal Predicts', edgecolor='black')

ax3.set_title('Extreme Events: Actual vs Normal', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Standard Deviations (|return| > x*std)', fontsize=10)
ax3.set_ylabel('Percentage of Days (%)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Time series with extreme events
ax4 = axes[1, 1]
dates = np.arange(len(returns_real))
ax4.plot(dates, returns_real, color=MLBLUE, alpha=0.5, linewidth=0.5)

# Highlight extreme events (> 3 std)
threshold = 3 * np.std(returns_real)
extreme_up = returns_real > threshold
extreme_down = returns_real < -threshold

ax4.scatter(dates[extreme_up], returns_real[extreme_up], color=MLGREEN, s=30, zorder=5, label='Extreme Up')
ax4.scatter(dates[extreme_down], returns_real[extreme_down], color=MLRED, s=30, zorder=5, label='Extreme Down')

ax4.axhline(threshold, color=MLGREEN, linestyle='--', linewidth=1.5, alpha=0.7)
ax4.axhline(-threshold, color=MLRED, linestyle='--', linewidth=1.5, alpha=0.7)
ax4.axhline(0, color='black', linewidth=1)

ax4.set_title(f'Extreme Events: {sum(extreme_up)+sum(extreme_down)} days > 3 std', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Trading Day', fontsize=10)
ax4.set_ylabel('Daily Return (%)', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Add expected count
normal_expected_count = len(returns_real) * 2 * (1 - stats.norm.cdf(3))
ax4.text(0.95, 0.95, f'Normal expects: {normal_expected_count:.0f} days\nActual: {sum(extreme_up)+sum(extreme_down)} days',
         transform=ax4.transAxes, fontsize=9, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
