"""Distribution Fitting - Finding the best fit"""
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
fig.suptitle('Distribution Fitting', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data that's not quite normal
data = np.concatenate([np.random.normal(0, 1, 800), np.random.normal(0, 3, 200)])

# Plot 1: Histogram with fitted distributions
ax1 = axes[0, 0]
ax1.hist(data, bins=40, density=True, alpha=0.5, color=MLBLUE, edgecolor='black', label='Data')

x = np.linspace(min(data), max(data), 200)

# Fit normal
mu, sigma = stats.norm.fit(data)
ax1.plot(x, stats.norm.pdf(x, mu, sigma), color=MLGREEN, linewidth=2, label=f'Normal (mu={mu:.2f}, sigma={sigma:.2f})')

# Fit t-distribution
df, loc, scale = stats.t.fit(data)
ax1.plot(x, stats.t.pdf(x, df, loc, scale), color=MLRED, linewidth=2, label=f't-dist (df={df:.1f})')

ax1.set_title('Fitting Distributions to Data', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: QQ plot comparison
ax2 = axes[0, 1]
# Custom QQ plot
sorted_data = np.sort(data)
n = len(data)
theoretical_q = np.array([(i - 0.5) / n for i in range(1, n + 1)])
normal_q = stats.norm.ppf(theoretical_q, mu, sigma)
t_q = stats.t.ppf(theoretical_q, df, loc, scale)

ax2.scatter(normal_q, sorted_data, alpha=0.5, s=10, color=MLGREEN, label='Normal fit')
ax2.scatter(t_q, sorted_data, alpha=0.5, s=10, color=MLRED, label='t-dist fit')
ax2.plot([min(data), max(data)], [min(data), max(data)], 'k--', linewidth=2)

ax2.set_title('QQ Plot: Which Fit is Better?', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Theoretical Quantiles', fontsize=10)
ax2.set_ylabel('Sample Quantiles', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Goodness of fit statistics
ax3 = axes[1, 0]

# KS test for each distribution
ks_norm = stats.kstest(data, 'norm', args=(mu, sigma))
ks_t = stats.kstest(data, 't', args=(df, loc, scale))

# Log-likelihood
ll_norm = np.sum(stats.norm.logpdf(data, mu, sigma))
ll_t = np.sum(stats.t.logpdf(data, df, loc, scale))

# AIC (simpler version)
aic_norm = -2 * ll_norm + 2 * 2  # 2 parameters
aic_t = -2 * ll_t + 2 * 3  # 3 parameters

metrics = ['KS Statistic', 'p-value', 'Log-Likelihood', 'AIC']
normal_vals = [ks_norm.statistic, ks_norm.pvalue, ll_norm, aic_norm]
t_vals = [ks_t.statistic, ks_t.pvalue, ll_t, aic_t]

x_pos = np.arange(len(metrics))
width = 0.35

ax3.bar(x_pos - width/2, normal_vals, width, color=MLGREEN, alpha=0.7, label='Normal', edgecolor='black')
ax3.bar(x_pos + width/2, t_vals, width, color=MLRED, alpha=0.7, label='t-dist', edgecolor='black')

ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics, fontsize=9)
ax3.set_title('Goodness of Fit Metrics', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, (n_val, t_val) in enumerate(zip(normal_vals, t_vals)):
    ax3.text(i - width/2, n_val + abs(n_val)*0.05, f'{n_val:.2f}', ha='center', fontsize=8)
    ax3.text(i + width/2, t_val + abs(t_val)*0.05, f'{t_val:.2f}', ha='center', fontsize=8)

# Plot 4: Fitting process
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Distribution Fitting Process', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

steps = [
    ('1. Visualize', 'ax.hist(data, density=True)', 'Look at shape: symmetric? skewed? fat tails?'),
    ('2. Fit candidates', 'mu, sigma = stats.norm.fit(data)', 'Estimate parameters using MLE'),
    ('3. Compare fits', 'stats.kstest(data, "norm")', 'KS test, AIC, visual QQ plots'),
    ('4. Validate', 'Check tail behavior', 'Are extreme events captured?'),
]

y = 0.78
for step, code, desc in steps:
    ax4.text(0.05, y, step, fontsize=11, fontweight='bold', color=MLBLUE, transform=ax4.transAxes)
    ax4.text(0.2, y, code, fontsize=9, family='monospace', transform=ax4.transAxes)
    ax4.text(0.2, y - 0.05, desc, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.18

# Winner box
winner = 't-distribution' if aic_t < aic_norm else 'Normal'
ax4.text(0.5, 0.08, f'Best fit: {winner} (lower AIC = better)',
         ha='center', fontsize=11, fontweight='bold', color=MLGREEN, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
