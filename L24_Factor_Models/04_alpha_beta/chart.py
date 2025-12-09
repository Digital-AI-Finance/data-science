"""Alpha and Beta - Skill vs Risk"""
import matplotlib.pyplot as plt
import numpy as np
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
fig.suptitle('Alpha and Beta: Separating Skill from Market Risk', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Alpha visualization
ax1 = axes[0, 0]

# Market returns
market_returns = np.random.normal(0.8, 4, 100)

# Stock with positive alpha
beta = 1.0
alpha = 0.3  # 0.3% per month excess return
stock_returns = alpha + beta * market_returns + np.random.normal(0, 2, 100)

ax1.scatter(market_returns, stock_returns, c=MLBLUE, s=50, alpha=0.6, edgecolors='black')

# Fit line
z = np.polyfit(market_returns, stock_returns, 1)
x_line = np.linspace(-10, 12, 100)
ax1.plot(x_line, np.poly1d(z)(x_line), color=MLRED, linewidth=2.5, label=f'Fitted: alpha={z[1]:.2f}%, beta={z[0]:.2f}')

# Zero-alpha line (pure beta)
ax1.plot(x_line, z[0] * x_line, color='gray', linewidth=1.5, linestyle='--', label='Zero alpha line')

# Highlight alpha
ax1.annotate('', xy=(0, z[1]), xytext=(0, 0),
             arrowprops=dict(arrowstyle='<->', color=MLGREEN, lw=2))
ax1.text(0.5, z[1]/2, f'Alpha\n{z[1]:.2f}%', fontsize=10, color=MLGREEN, fontweight='bold')

ax1.axhline(0, color='gray', linewidth=1, linestyle=':')
ax1.axvline(0, color='gray', linewidth=1, linestyle=':')

ax1.set_title('Alpha: Return Above What Beta Explains', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Market Return (%)', fontsize=10)
ax1.set_ylabel('Stock Return (%)', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Alpha distribution across funds
ax2 = axes[0, 1]

# Simulated fund alphas (annualized)
n_funds = 500
fund_alphas = np.random.normal(-0.5, 2, n_funds)  # Most funds have negative alpha

ax2.hist(fund_alphas, bins=40, color=MLBLUE, edgecolor='black', alpha=0.7)
ax2.axvline(0, color=MLRED, linewidth=2.5, linestyle='--', label='Zero Alpha')
ax2.axvline(np.mean(fund_alphas), color=MLORANGE, linewidth=2.5, label=f'Mean: {np.mean(fund_alphas):.2f}%')

positive_alpha = np.sum(fund_alphas > 0) / len(fund_alphas) * 100

ax2.set_title('Alpha Distribution: 500 Mutual Funds (Annualized)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Alpha (%)', fontsize=10)
ax2.set_ylabel('Number of Funds', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis='y')

ax2.text(0.95, 0.95, f'Only {positive_alpha:.1f}% have\npositive alpha',
         transform=ax2.transAxes, fontsize=10, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 3: Beta spectrum
ax3 = axes[1, 0]

assets = ['Treasury\nBonds', 'Utilities', 'Consumer\nStaples', 'S&P 500', 'Financials', 'Tech', 'Bitcoin']
betas = [0.05, 0.45, 0.65, 1.0, 1.25, 1.35, 2.5]

colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(assets)))
bars = ax3.bar(assets, betas, color=colors_gradient, edgecolor='black', linewidth=0.5)

ax3.axhline(1.0, color=MLRED, linewidth=2, linestyle='--', label='Market Beta = 1')
ax3.axhline(0, color='gray', linewidth=1)

ax3.set_title('Beta Spectrum: Defensive to Aggressive', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Beta', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')

# Add zone labels
ax3.text(-0.5, 0.3, 'Defensive\n(Beta < 1)', fontsize=9, style='italic', color=MLGREEN)
ax3.text(4.5, 1.8, 'Aggressive\n(Beta > 1)', fontsize=9, style='italic', color=MLRED)

# Plot 4: Summary formulas
ax4 = axes[1, 1]
ax4.axis('off')

summary = r'''
ALPHA VS BETA: KEY CONCEPTS

BETA (Risk Exposure)
--------------------
- Measures sensitivity to market
- Cannot be "generated" - it's exposure
- High beta = high risk, high expected return
- Can be achieved cheaply via index funds

ALPHA (Skill Premium)
---------------------
- Return AFTER accounting for risk factors
- Represents manager skill (or luck)
- Very difficult to generate consistently
- Most active managers have negative alpha

THE MATH:
$R_i - R_f = \alpha + \beta \cdot (R_m - R_f) + \epsilon$

Rearranging:
$\alpha = (R_i - R_f) - \beta \cdot (R_m - R_f)$

EXAMPLE (Annualized):
- Fund return: 12%
- Risk-free: 2%
- Market return: 10%
- Fund beta: 1.2

Alpha = (12% - 2%) - 1.2 * (10% - 2%)
Alpha = 10% - 9.6% = 0.4%

The fund added 0.4% above its risk-adjusted benchmark.
'''

ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Alpha vs Beta Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
