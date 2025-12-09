"""Multi-Factor Models - Beyond Fama-French"""
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
fig.suptitle('Multi-Factor Models: Comprehensive Risk Decomposition', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Factor zoo overview
ax1 = axes[0, 0]
ax1.axis('off')

factor_zoo = '''
THE FACTOR ZOO

Academic Factors (Well-Established)
-----------------------------------
Market (MKT)    : Equity premium
Size (SMB)      : Small cap premium
Value (HML)     : Value vs growth
Momentum (MOM)  : Winners keep winning
Profitability   : Profitable > unprofitable
Investment      : Conservative > aggressive
Quality         : Low debt, stable earnings

Industry Factors (Practitioner)
-------------------------------
BAB             : Betting Against Beta
QMJ             : Quality Minus Junk
LIQ             : Liquidity risk premium
VOL             : Volatility factor
TERM            : Term structure
CREDIT          : Credit spread

THE PROBLEM: Factor proliferation
- 400+ published factors
- Many don't replicate
- Data mining concerns
- Use skeptically!
'''

ax1.text(0.02, 0.98, factor_zoo, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('The Factor Zoo', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Model comparison - R-squared
ax2 = axes[0, 1]

models = ['CAPM', 'FF3', 'Carhart 4', 'FF5', 'FF6', 'Custom 10']
avg_rsq = [0.62, 0.73, 0.76, 0.78, 0.79, 0.85]
n_factors = [1, 3, 4, 5, 6, 10]

# Create bar chart
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(models)))
bars = ax2.bar(models, avg_rsq, color=colors, edgecolor='black', linewidth=0.5)

ax2.set_title('Model Explanatory Power (Cross-sectional R-sq)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Average R-squared', fontsize=10)
ax2.set_ylim(0.5, 1.0)
ax2.grid(alpha=0.3, axis='y')

# Add factor counts
for bar, r2, n in zip(bars, avg_rsq, n_factors):
    ax2.text(bar.get_x() + bar.get_width()/2, r2 + 0.01, f'{r2:.0%}\n({n}F)',
             ha='center', fontsize=9)

# Plot 3: Multi-factor regression output
ax3 = axes[1, 0]

factors = ['Alpha', 'MKT', 'SMB', 'HML', 'MOM', 'RMW', 'CMA']
coefs = [0.12, 1.08, 0.35, -0.22, 0.15, 0.18, -0.08]
std_errs = [0.08, 0.05, 0.08, 0.07, 0.06, 0.09, 0.07]

# Calculate t-stats
t_stats = [c/s for c, s in zip(coefs, std_errs)]
significant = [abs(t) > 2 for t in t_stats]
colors = [MLGREEN if sig else MLRED for sig in significant]

y_pos = np.arange(len(factors))
ax3.barh(y_pos, coefs, xerr=np.array(std_errs)*2, color=colors,
         edgecolor='black', linewidth=0.5, capsize=3)

ax3.axvline(0, color='gray', linewidth=1.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(factors)
ax3.set_title('Fama-French 6-Factor Model Output', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Coefficient (with 95% CI)', fontsize=10)
ax3.grid(alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MLGREEN, label='Significant (|t|>2)'),
                   Patch(facecolor=MLRED, label='Not significant')]
ax3.legend(handles=legend_elements, fontsize=8, loc='lower right')

# Plot 4: Code implementation
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Multi-Factor Model in sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Prepare factor data
factors = ['MKT', 'SMB', 'HML', 'MOM', 'RMW', 'CMA']
X = ff_data[factors]
y = stock_excess_returns

# Option 1: sklearn (simple)
model = LinearRegression()
model.fit(X, y)
print(f"Alpha: {model.intercept_:.4f}")
print(f"Betas: {dict(zip(factors, model.coef_))}")
print(f"R-squared: {r2_score(y, model.predict(X)):.4f}")

# Option 2: statsmodels (with t-stats)
X_sm = sm.add_constant(X)  # Add intercept
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())

# Key outputs from statsmodels:
# - Coefficients with standard errors
# - t-statistics and p-values
# - R-squared and Adjusted R-squared
# - F-statistic for overall significance
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
