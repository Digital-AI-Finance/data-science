"""Factor Loadings - Interpreting betas"""
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
fig.suptitle('Factor Loadings: Interpreting Beta Coefficients', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Factor loadings comparison across stocks
ax1 = axes[0, 0]

stocks = ['Tech ETF\n(QQQ)', 'Value ETF\n(VTV)', 'Small Cap\n(IWM)', 'Bond ETF\n(BND)', 'Gold ETF\n(GLD)']
mkt_betas = [1.15, 0.95, 1.20, 0.05, 0.10]
smb_betas = [-0.15, -0.10, 0.85, -0.02, -0.05]
hml_betas = [-0.45, 0.60, 0.25, 0.02, -0.15]

x = np.arange(len(stocks))
width = 0.25

bars1 = ax1.bar(x - width, mkt_betas, width, label='MKT Beta', color=MLBLUE, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x, smb_betas, width, label='SMB Beta', color=MLGREEN, edgecolor='black', linewidth=0.5)
bars3 = ax1.bar(x + width, hml_betas, width, label='HML Beta', color=MLORANGE, edgecolor='black', linewidth=0.5)

ax1.axhline(0, color='gray', linewidth=1.5)
ax1.set_xticks(x)
ax1.set_xticklabels(stocks, fontsize=9)
ax1.set_title('Factor Loadings by Asset Type', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Factor Beta', fontsize=10)
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(alpha=0.3, axis='y')

# Plot 2: What betas mean
ax2 = axes[0, 1]
ax2.axis('off')

interpretation = '''
INTERPRETING FACTOR LOADINGS

MKT Beta (Market Sensitivity)
-----------------------------
Beta > 1: More volatile than market (aggressive)
Beta = 1: Moves with market
Beta < 1: Less volatile (defensive)
Beta ~ 0: Uncorrelated with market

SMB Beta (Size Exposure)
------------------------
Beta > 0: Behaves like small caps
Beta < 0: Behaves like large caps
Beta ~ 0: No size tilt

HML Beta (Value Exposure)
-------------------------
Beta > 0: Behaves like value stocks
Beta < 0: Behaves like growth stocks
Beta ~ 0: Blend

EXAMPLE: Tech ETF (QQQ)
- MKT = 1.15: Slightly more volatile
- SMB = -0.15: Large cap tilt
- HML = -0.45: Strong growth tilt

EXAMPLE: Small Cap ETF (IWM)
- MKT = 1.20: More volatile
- SMB = 0.85: Strong small cap exposure
- HML = 0.25: Slight value tilt
'''

ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Beta Interpretation Guide', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Beta estimation over time (rolling)
ax3 = axes[1, 0]

months = 60
dates = np.arange(months)

# Rolling 12-month beta estimates
rolling_mkt = 1.0 + 0.3 * np.sin(dates/10) + np.random.normal(0, 0.1, months)
rolling_smb = -0.2 + 0.2 * np.cos(dates/15) + np.random.normal(0, 0.08, months)
rolling_hml = -0.4 + 0.15 * np.sin(dates/8) + np.random.normal(0, 0.06, months)

ax3.plot(dates, rolling_mkt, color=MLBLUE, linewidth=2, label='MKT Beta')
ax3.plot(dates, rolling_smb, color=MLGREEN, linewidth=2, label='SMB Beta')
ax3.plot(dates, rolling_hml, color=MLORANGE, linewidth=2, label='HML Beta')

ax3.axhline(0, color='gray', linewidth=1, linestyle='--')
ax3.fill_between(dates, rolling_mkt - 0.2, rolling_mkt + 0.2, alpha=0.2, color=MLBLUE)

ax3.set_title('Rolling 12-Month Factor Betas', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Month', fontsize=10)
ax3.set_ylabel('Beta Estimate', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

ax3.text(0.02, 0.02, 'Shaded area: 95% confidence interval for MKT beta',
         transform=ax3.transAxes, fontsize=8, style='italic')

# Plot 4: Statistical significance
ax4 = axes[1, 1]

factors = ['MKT', 'SMB', 'HML', 'MOM', 'Alpha']
betas = [1.15, 0.25, -0.35, 0.12, 0.08]
t_stats = [12.5, 2.8, -3.2, 1.1, 0.5]
significant = [abs(t) > 2 for t in t_stats]

colors = [MLGREEN if sig else MLRED for sig in significant]
bars = ax4.barh(factors, t_stats, color=colors, edgecolor='black', linewidth=0.5)

ax4.axvline(-2, color='gray', linewidth=2, linestyle='--', label='Significance threshold (|t|=2)')
ax4.axvline(2, color='gray', linewidth=2, linestyle='--')
ax4.axvline(0, color='black', linewidth=1)

ax4.set_title('Statistical Significance of Factor Loadings', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('t-statistic', fontsize=10)
ax4.legend(fontsize=8, loc='lower right')
ax4.grid(alpha=0.3, axis='x')

# Add annotations
for bar, t, b in zip(bars, t_stats, betas):
    label = f't={t:.1f}, beta={b:.2f}'
    x_pos = t + 0.5 if t > 0 else t - 0.5
    ha = 'left' if t > 0 else 'right'
    ax4.text(x_pos, bar.get_y() + bar.get_height()/2, label,
             va='center', ha=ha, fontsize=8)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
