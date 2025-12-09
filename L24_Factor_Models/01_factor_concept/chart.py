"""Factor Model Concept - What is a factor?"""
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
fig.suptitle('Factor Models: Understanding Return Drivers', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Single factor model (CAPM)
ax1 = axes[0, 0]

# Market returns vs stock returns
n = 100
market_returns = np.random.normal(0.8, 3, n)
beta = 1.2
alpha = 0.1
stock_returns = alpha + beta * market_returns + np.random.normal(0, 1.5, n)

ax1.scatter(market_returns, stock_returns, c=MLBLUE, s=50, alpha=0.6, edgecolors='black')

# Fit line
z = np.polyfit(market_returns, stock_returns, 1)
x_line = np.linspace(market_returns.min(), market_returns.max(), 100)
ax1.plot(x_line, np.poly1d(z)(x_line), color=MLRED, linewidth=2.5, label=f'Beta = {z[0]:.2f}')

ax1.axhline(0, color='gray', linewidth=1, linestyle='--')
ax1.axvline(0, color='gray', linewidth=1, linestyle='--')

ax1.set_title('Single Factor: CAPM', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Market Return (%)', fontsize=10)
ax1.set_ylabel('Stock Return (%)', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Factor decomposition
ax2 = axes[0, 1]

factors = ['Market\n(Beta)', 'Size\n(SMB)', 'Value\n(HML)', 'Momentum\n(MOM)', 'Alpha\n(Skill)']
contributions = [4.2, 1.1, -0.5, 0.8, 0.3]
colors = [MLBLUE, MLGREEN, MLRED, MLORANGE, MLPURPLE]

bars = ax2.bar(factors, contributions, color=colors, edgecolor='black', linewidth=0.5)
ax2.axhline(0, color='gray', linewidth=1.5)

# Add total return annotation
total = sum(contributions)
ax2.axhline(total, color='black', linestyle='--', linewidth=2, label=f'Total Return: {total:.1f}%')

ax2.set_title('Return Decomposition by Factor', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Return Contribution (%)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, contributions):
    y_pos = val + 0.1 if val > 0 else val - 0.3
    ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.1f}%',
             ha='center', fontsize=10, fontweight='bold')

# Plot 3: Factor model equation
ax3 = axes[1, 0]
ax3.axis('off')

equation = r'''
FACTOR MODEL EQUATION

$R_i = \alpha + \beta_1 F_1 + \beta_2 F_2 + ... + \beta_k F_k + \epsilon$

Where:
- R_i = Return of asset i
- alpha = Unexplained return (skill)
- beta_k = Sensitivity to factor k
- F_k = Return of factor k
- epsilon = Idiosyncratic noise

SINGLE FACTOR (CAPM):
$R_i - R_f = \alpha + \beta (R_m - R_f) + \epsilon$

MULTI-FACTOR (Fama-French):
$R_i - R_f = \alpha + \beta_1 MKT + \beta_2 SMB + \beta_3 HML + \epsilon$

Key Insight:
- Factors explain SYSTEMATIC return sources
- Alpha is the unexplained residual
- Good models have high R-sq and low alpha
'''

ax3.text(0.02, 0.95, equation, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Factor Model Equations', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: R-squared progression
ax4 = axes[1, 1]

models = ['CAPM\n(1 factor)', 'FF3\n(3 factors)', 'FF5\n(5 factors)', 'Custom\n(10 factors)']
r_squared = [0.65, 0.78, 0.82, 0.88]

bars = ax4.bar(models, r_squared, color=[MLBLUE, MLORANGE, MLGREEN, MLPURPLE],
               edgecolor='black', linewidth=0.5)

ax4.set_title('Explained Variance by Model Complexity', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_ylabel('R-squared', fontsize=10)
ax4.set_ylim(0.5, 1.0)
ax4.grid(alpha=0.3, axis='y')

# Add values
for bar, r2 in zip(bars, r_squared):
    ax4.text(bar.get_x() + bar.get_width()/2, r2 + 0.01, f'{r2:.0%}',
             ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
