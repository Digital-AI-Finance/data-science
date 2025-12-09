"""Portfolio Factor Analysis - Real-world application"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
fig.suptitle('Portfolio Factor Analysis: Real-World Application', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Portfolio decomposition
ax1 = axes[0, 0]

# Sample portfolio factor exposures
portfolios = ['Growth\nPortfolio', 'Value\nPortfolio', 'Balanced\nPortfolio', 'Factor\nNeutral']
mkt = [1.15, 0.95, 1.02, 1.00]
smb = [-0.2, 0.15, 0.05, 0.00]
hml = [-0.5, 0.55, 0.10, 0.00]
mom = [0.25, -0.10, 0.08, 0.00]

x = np.arange(len(portfolios))
width = 0.2

ax1.bar(x - 1.5*width, mkt, width, label='MKT', color=MLBLUE, edgecolor='black', linewidth=0.5)
ax1.bar(x - 0.5*width, smb, width, label='SMB', color=MLGREEN, edgecolor='black', linewidth=0.5)
ax1.bar(x + 0.5*width, hml, width, label='HML', color=MLORANGE, edgecolor='black', linewidth=0.5)
ax1.bar(x + 1.5*width, mom, width, label='MOM', color=MLPURPLE, edgecolor='black', linewidth=0.5)

ax1.axhline(0, color='gray', linewidth=1.5)
ax1.set_xticks(x)
ax1.set_xticklabels(portfolios, fontsize=9)
ax1.set_title('Portfolio Factor Exposures', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Factor Beta', fontsize=10)
ax1.legend(fontsize=8, ncol=4, loc='upper center')
ax1.grid(alpha=0.3, axis='y')

# Plot 2: Return attribution
ax2 = axes[1, 0]

# Attribution for a portfolio
factors_attr = ['MKT\nContrib.', 'SMB\nContrib.', 'HML\nContrib.', 'MOM\nContrib.', 'Alpha', 'Total\nReturn']
contributions = [6.5, 0.8, -1.2, 1.5, 0.4, 8.0]
colors = [MLBLUE, MLGREEN, MLRED, MLORANGE, MLPURPLE, 'black']

bars = ax2.bar(factors_attr, contributions, color=colors, edgecolor='black', linewidth=0.5)
ax2.axhline(0, color='gray', linewidth=1.5)

# Connect to total
ax2.plot([0, 1, 2, 3, 4], [6.5, 6.5+0.8, 6.5+0.8-1.2, 6.5+0.8-1.2+1.5, 8.0],
         color='black', linewidth=1.5, linestyle='--', marker='o', markersize=4)

ax2.set_title('Return Attribution (Growth Portfolio, Annual)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Return (%)', fontsize=10)
ax2.grid(alpha=0.3, axis='y')

# Add values
for bar, val in zip(bars, contributions):
    y_pos = val + 0.2 if val > 0 else val - 0.4
    ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.1f}%',
             ha='center', fontsize=10, fontweight='bold')

# Plot 3: Risk contribution pie chart
ax3 = axes[0, 1]

# Risk contributions (variance decomposition)
risk_sources = ['Market Risk', 'Size Risk', 'Value Risk', 'Momentum Risk', 'Idiosyncratic']
risk_pcts = [55, 12, 18, 8, 7]
colors_pie = [MLBLUE, MLGREEN, MLORANGE, MLPURPLE, 'gray']

wedges, texts, autotexts = ax3.pie(risk_pcts, labels=risk_sources, autopct='%1.0f%%',
                                    colors=colors_pie, explode=[0.05, 0, 0, 0, 0],
                                    textprops={'fontsize': 9})

ax3.set_title('Risk Decomposition (Variance Attribution)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete analysis workflow
ax4 = axes[1, 1]
ax4.axis('off')

workflow = '''
COMPLETE FACTOR ANALYSIS WORKFLOW

1. LOAD PORTFOLIO HOLDINGS
   holdings = pd.read_csv('portfolio.csv')
   # ticker, weight

2. CALCULATE PORTFOLIO RETURNS
   portfolio_ret = (holdings['weight'] *
                    stock_returns).sum(axis=1)

3. REGRESS ON FACTORS
   from sklearn.linear_model import LinearRegression

   X = ff_factors[['Mkt-RF', 'SMB', 'HML', 'MOM']]
   y = portfolio_ret - ff_factors['RF']

   model = LinearRegression()
   model.fit(X, y)

4. ANALYZE RESULTS
   print("Factor Exposures:")
   print(f"  MKT: {model.coef_[0]:.3f}")
   print(f"  SMB: {model.coef_[1]:.3f}")
   print(f"  HML: {model.coef_[2]:.3f}")
   print(f"  MOM: {model.coef_[3]:.3f}")
   print(f"  Alpha: {model.intercept_:.3f}")

5. CALCULATE ATTRIBUTIONS
   factor_returns = X.mean()
   contributions = model.coef_ * factor_returns

6. REPORT TO STAKEHOLDERS
   - Which factors drove performance?
   - Is alpha statistically significant?
   - Are factor exposures intentional?
'''

ax4.text(0.02, 0.98, workflow, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Factor Analysis Workflow', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
