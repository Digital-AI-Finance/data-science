"""Math Functions - NumPy mathematical functions"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NumPy Mathematical Functions', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: np.exp and np.log - Compound returns
ax1 = axes[0, 0]
np.random.seed(42)
daily_returns = np.random.normal(0.0005, 0.015, 252)
log_returns = np.log(1 + daily_returns)

cumulative_simple = np.cumprod(1 + daily_returns)
cumulative_log = np.exp(np.cumsum(log_returns))

ax1.plot(cumulative_simple, color=MLBLUE, linewidth=2, label='Simple: cumprod(1+r)')
ax1.plot(cumulative_log, color=MLORANGE, linewidth=2, linestyle='--', label='Log: exp(cumsum(log(1+r)))')

ax1.set_xlabel('Trading Day', fontsize=10)
ax1.set_ylabel('Growth Factor', fontsize=10)
ax1.set_title('np.exp() and np.log() for Returns', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: np.sqrt for volatility
ax2 = axes[0, 1]
daily_vol = np.std(daily_returns)
periods = [1, 5, 21, 63, 126, 252]
annualized_vol = [daily_vol * np.sqrt(p) * 100 for p in periods]
labels = ['1 Day', '1 Week', '1 Month', '1 Qtr', '6 Mo', '1 Year']

ax2.bar(labels, annualized_vol, color=MLGREEN, alpha=0.7, edgecolor='black')
ax2.plot(labels, annualized_vol, 'ro-', markersize=8)

ax2.set_xlabel('Time Period', fontsize=10)
ax2.set_ylabel('Volatility (%)', fontsize=10)
ax2.set_title('np.sqrt() for Volatility Scaling', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(axis='y', alpha=0.3)

# Add formula
ax2.text(0.5, 0.95, r'Annual Vol = Daily Vol $\times$ $\sqrt{252}$',
         transform=ax2.transAxes, fontsize=10, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

# Plot 3: np.percentile for risk metrics
ax3 = axes[1, 0]
returns_sorted = np.sort(daily_returns) * 100
percentiles = [5, 25, 50, 75, 95]
percentile_vals = [np.percentile(daily_returns * 100, p) for p in percentiles]

ax3.hist(daily_returns * 100, bins=30, color=MLBLUE, alpha=0.5, edgecolor='black', density=True)

for p, val in zip(percentiles, percentile_vals):
    color = MLRED if p == 5 else MLGREEN if p == 95 else MLPURPLE
    ax3.axvline(val, color=color, linestyle='--', linewidth=2, label=f'{p}th: {val:.2f}%')

ax3.set_xlabel('Daily Return (%)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title('np.percentile() for Risk Analysis', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(alpha=0.3)

# VaR annotation
ax3.annotate('VaR 95%', xy=(percentile_vals[0], 0.1), xytext=(percentile_vals[0]-2, 0.2),
            arrowprops=dict(arrowstyle='->', color=MLRED), fontsize=9, color=MLRED)

# Plot 4: np.clip for outlier handling
ax4 = axes[1, 1]
# Generate data with outliers
returns_with_outliers = daily_returns.copy() * 100
returns_with_outliers[50] = 15  # Outlier
returns_with_outliers[150] = -12  # Outlier

clipped = np.clip(returns_with_outliers, -5, 5)

ax4.plot(returns_with_outliers, color=MLRED, alpha=0.5, linewidth=1, label='Original (with outliers)')
ax4.plot(clipped, color=MLGREEN, alpha=0.8, linewidth=1, label='np.clip(-5%, 5%)')
ax4.axhline(5, color='gray', linestyle='--', alpha=0.5)
ax4.axhline(-5, color='gray', linestyle='--', alpha=0.5)

ax4.set_xlabel('Trading Day', fontsize=10)
ax4.set_ylabel('Return (%)', fontsize=10)
ax4.set_title('np.clip() for Outlier Handling', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
