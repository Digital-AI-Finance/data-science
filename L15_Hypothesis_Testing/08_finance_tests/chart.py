"""Finance-Specific Hypothesis Tests"""
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
fig.suptitle('Hypothesis Tests in Finance', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Testing for alpha (excess return)
ax1 = axes[0, 0]
n = 252
rf = 0.01 / 252  # Daily risk-free rate
market_returns = np.random.normal(0.0004, 0.01, n)
fund_returns = market_returns + np.random.normal(0.0002, 0.005, n)  # Small alpha

# Calculate alpha (simplified)
excess_returns = fund_returns - rf
t_stat, p_value = stats.ttest_1samp(excess_returns, 0)

ax1.hist(excess_returns * 100, bins=30, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')
ax1.axvline(np.mean(excess_returns) * 100, color=MLGREEN, linewidth=2.5,
            label=f'Mean alpha = {np.mean(excess_returns)*252*100:.2f}% (ann.)')
ax1.axvline(0, color=MLRED, linewidth=2, linestyle='--', label='H0: alpha = 0')

ax1.set_title(f'Testing for Alpha (t = {t_stat:.2f}, p = {p_value:.4f})', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Daily Excess Return (%)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: F-test for variance comparison
ax2 = axes[0, 1]
low_vol = np.random.normal(0, 1.0, 100)
high_vol = np.random.normal(0, 1.5, 100)

# F-test
f_stat = np.var(high_vol, ddof=1) / np.var(low_vol, ddof=1)
p_value_f = 1 - stats.f.cdf(f_stat, len(high_vol)-1, len(low_vol)-1)

bins = np.linspace(-5, 5, 30)
ax2.hist(low_vol, bins=bins, density=True, alpha=0.5, color=MLBLUE, label=f'Low Vol (std={np.std(low_vol):.2f})')
ax2.hist(high_vol, bins=bins, density=True, alpha=0.5, color=MLRED, label=f'High Vol (std={np.std(high_vol):.2f})')

ax2.set_title(f'F-test for Variance (F = {f_stat:.2f}, p = {p_value_f:.4f})', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Return (%)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Jarque-Bera test for normality
ax3 = axes[1, 0]
returns_nonnormal = np.concatenate([np.random.normal(0, 1, 950), np.random.normal(0, 3, 50)])

# JB test
jb_stat, jb_p = stats.jarque_bera(returns_nonnormal)

ax3.hist(returns_nonnormal, bins=40, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')

# Overlay normal
x = np.linspace(min(returns_nonnormal), max(returns_nonnormal), 100)
ax3.plot(x, stats.norm.pdf(x, np.mean(returns_nonnormal), np.std(returns_nonnormal)),
         color=MLRED, linewidth=2.5, label='Normal fit')

skew = stats.skew(returns_nonnormal)
kurt = stats.kurtosis(returns_nonnormal) + 3

ax3.set_title(f'Jarque-Bera: JB = {jb_stat:.1f}, p = {jb_p:.4f}', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Return', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.text(0.95, 0.95, f'Skew = {skew:.2f}\nKurt = {kurt:.2f}', transform=ax3.transAxes,
         fontsize=10, va='top', ha='right', bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))

# Plot 4: Summary of finance tests
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Common Finance Hypothesis Tests', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

tests = [
    ('t-test', 'Test for alpha (excess returns)', 'stats.ttest_1samp()', MLBLUE),
    ('F-test', 'Compare variances (volatility)', 'stats.f_oneway()', MLGREEN),
    ('Jarque-Bera', 'Test for normality', 'stats.jarque_bera()', MLORANGE),
    ('Ljung-Box', 'Test for autocorrelation', 'acorr_ljungbox()', MLRED),
    ('ADF Test', 'Test for stationarity', 'adfuller()', MLPURPLE),
    ('ARCH Test', 'Test for heteroskedasticity', 'het_arch()', MLBLUE),
]

y = 0.78
for name, use, code, color in tests:
    ax4.text(0.05, y, name + ':', fontsize=10, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.25, y, use, fontsize=9, transform=ax4.transAxes)
    ax4.text(0.65, y, code, fontsize=8, family='monospace', color='gray', transform=ax4.transAxes)
    y -= 0.11

ax4.text(0.5, 0.08, 'Always check assumptions before applying any test!',
         ha='center', fontsize=10, style='italic', color=MLRED, transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
