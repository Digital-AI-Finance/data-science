"""PDF and CDF - Probability and Cumulative Functions"""
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
fig.suptitle('PDF and CDF: Probability Functions', fontsize=14, fontweight='bold', color=MLPURPLE)

x = np.linspace(-4, 4, 200)

# Plot 1: PDF explained
ax1 = axes[0, 0]
pdf = stats.norm.pdf(x)
ax1.plot(x, pdf, color=MLBLUE, linewidth=2.5, label='PDF')

# Shade area
x_fill = np.linspace(0.5, 1.5, 100)
pdf_fill = stats.norm.pdf(x_fill)
ax1.fill_between(x_fill, pdf_fill, alpha=0.5, color=MLORANGE)

# Calculate probability
prob = stats.norm.cdf(1.5) - stats.norm.cdf(0.5)
ax1.annotate(f'P(0.5 < X < 1.5)\n= {prob:.3f}', xy=(1, 0.15), fontsize=10,
             ha='center', color=MLORANGE, fontweight='bold')

ax1.set_title('PDF: Probability Density Function', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Value (x)', fontsize=10)
ax1.set_ylabel('Density f(x)', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, 'Area under curve = Probability', transform=ax1.transAxes,
         fontsize=9, va='top', style='italic')

# Plot 2: CDF explained
ax2 = axes[0, 1]
cdf = stats.norm.cdf(x)
ax2.plot(x, cdf, color=MLGREEN, linewidth=2.5, label='CDF')

# Mark key points
for val in [-1, 0, 1]:
    cdf_val = stats.norm.cdf(val)
    ax2.scatter([val], [cdf_val], color=MLRED, s=80, zorder=5)
    ax2.hlines(cdf_val, -4, val, colors=MLRED, linestyles='--', alpha=0.5)
    ax2.vlines(val, 0, cdf_val, colors=MLRED, linestyles='--', alpha=0.5)
    ax2.annotate(f'{cdf_val:.2f}', xy=(val, cdf_val + 0.05), fontsize=9, ha='center')

ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1)
ax2.set_title('CDF: Cumulative Distribution Function', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Value (x)', fontsize=10)
ax2.set_ylabel('F(x) = P(X <= x)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.text(0.05, 0.95, 'CDF(x) = Probability X <= x', transform=ax2.transAxes,
         fontsize=9, va='top', style='italic')

# Plot 3: PDF vs CDF side by side
ax3 = axes[1, 0]
ax3_twin = ax3.twinx()

ax3.plot(x, pdf, color=MLBLUE, linewidth=2.5, label='PDF')
ax3.fill_between(x, pdf, alpha=0.2, color=MLBLUE)
ax3_twin.plot(x, cdf, color=MLGREEN, linewidth=2.5, label='CDF')

ax3.set_xlabel('Value (x)', fontsize=10)
ax3.set_ylabel('PDF - Density', fontsize=10, color=MLBLUE)
ax3_twin.set_ylabel('CDF - Cumulative Prob', fontsize=10, color=MLGREEN)
ax3.tick_params(axis='y', colors=MLBLUE)
ax3_twin.tick_params(axis='y', colors=MLGREEN)

ax3.set_title('PDF and CDF Together', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(loc='upper left', fontsize=9)
ax3_twin.legend(loc='center right', fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Finance application - VaR
ax4 = axes[1, 1]
# Stock returns distribution
returns_x = np.linspace(-10, 10, 200)
returns_pdf = stats.norm.pdf(returns_x, 0, 2)  # Mean 0, Std 2

ax4.plot(returns_x, returns_pdf, color=MLBLUE, linewidth=2.5)

# VaR at 95% confidence (5th percentile)
var_95 = stats.norm.ppf(0.05, 0, 2)
ax4.fill_between(returns_x, returns_pdf, where=returns_x <= var_95,
                 alpha=0.5, color=MLRED, label=f'5% tail (VaR = {var_95:.2f}%)')
ax4.axvline(var_95, color=MLRED, linewidth=2, linestyle='--')

# VaR at 99%
var_99 = stats.norm.ppf(0.01, 0, 2)
ax4.axvline(var_99, color=MLORANGE, linewidth=2, linestyle='--', label=f'1% VaR = {var_99:.2f}%')

ax4.set_title('VaR: Using CDF Inverse (Quantile)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Daily Return (%)', fontsize=10)
ax4.set_ylabel('Probability Density', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

ax4.text(0.95, 0.95, 'VaR = stats.norm.ppf(0.05, mu, sigma)',
         transform=ax4.transAxes, fontsize=9, va='top', ha='right', family='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
