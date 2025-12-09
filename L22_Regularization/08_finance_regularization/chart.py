"""Finance Regularization - Practical applications"""
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
fig.suptitle('Regularization in Finance Applications', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Factor model - OLS overfits with many factors
ax1 = axes[0, 0]

factors = ['Mkt', 'SMB', 'HML', 'Mom', 'Qual', 'Vol', 'Liq', 'Sent', 'Macro1', 'Macro2']
n_factors = len(factors)

# In-sample (overfits)
ols_insample_r2 = 0.45
ols_outsample_r2 = 0.15

ridge_insample_r2 = 0.38
ridge_outsample_r2 = 0.32

lasso_insample_r2 = 0.35
lasso_outsample_r2 = 0.33

x = np.arange(3)
width = 0.35

in_sample = [ols_insample_r2, ridge_insample_r2, lasso_insample_r2]
out_sample = [ols_outsample_r2, ridge_outsample_r2, lasso_outsample_r2]

bars1 = ax1.bar(x - width/2, in_sample, width, label='In-Sample R-sq', color=MLBLUE, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, out_sample, width, label='Out-of-Sample R-sq', color=MLORANGE, edgecolor='black', linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(['OLS\n(10 factors)', 'Ridge\n(10 factors)', 'Lasso\n(6 factors)'])
ax1.set_title('Factor Model Performance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('R-squared', fontsize=10)
ax1.legend(fontsize=8)
ax1.set_ylim(0, 0.55)
ax1.grid(alpha=0.3, axis='y')

# Add gap annotations
for i, (ins, outs) in enumerate(zip(in_sample, out_sample)):
    gap = ins - outs
    ax1.annotate(f'Gap: {gap:.2f}', xy=(i, ins), xytext=(i + 0.15, ins + 0.03),
                fontsize=8, color=MLRED if gap > 0.15 else MLGREEN)

# Plot 2: Stock prediction - regularization improves out-of-sample
ax2 = axes[0, 1]

# Monthly predictions over 36 months
months = np.arange(1, 37)

# OLS: high variance, poor performance
ols_pred_corr = 0.02 + np.random.normal(0, 0.08, 36)
ols_pred_corr = np.clip(ols_pred_corr, -0.15, 0.25)

# Ridge: more stable
ridge_pred_corr = 0.08 + np.random.normal(0, 0.04, 36)
ridge_pred_corr = np.clip(ridge_pred_corr, -0.05, 0.20)

ax2.plot(months, ols_pred_corr, color=MLBLUE, linewidth=1.5, alpha=0.7, label=f'OLS (mean: {np.mean(ols_pred_corr):.3f})')
ax2.plot(months, ridge_pred_corr, color=MLGREEN, linewidth=2, label=f'Ridge (mean: {np.mean(ridge_pred_corr):.3f})')

ax2.axhline(0, color='gray', linewidth=1, linestyle='--')
ax2.fill_between(months, 0, ols_pred_corr, where=ols_pred_corr < 0, alpha=0.3, color=MLRED)

ax2.set_title('Monthly Prediction Correlation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Month', fontsize=10)
ax2.set_ylabel('Correlation (pred vs actual)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Portfolio optimization - regularized weights
ax3 = axes[1, 0]

assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'V', 'JNJ', 'PG', 'XOM', 'CVX']

# Unconstrained MVO weights (extreme)
mvo_weights = np.array([0.35, -0.15, 0.25, 0.40, -0.20, 0.15, 0.10, -0.10, 0.30, -0.10])

# Regularized weights (more balanced)
reg_weights = np.array([0.15, 0.05, 0.12, 0.18, 0.08, 0.10, 0.12, 0.08, 0.07, 0.05])

x = np.arange(len(assets))
width = 0.35

bars1 = ax3.bar(x - width/2, mvo_weights, width, label='MVO (unconstrained)',
                color=[MLGREEN if w > 0 else MLRED for w in mvo_weights], edgecolor='black', linewidth=0.5)
bars2 = ax3.bar(x + width/2, reg_weights, width, label='Regularized', color=MLBLUE, edgecolor='black', linewidth=0.5)

ax3.axhline(0, color='gray', linewidth=1)
ax3.set_xticks(x)
ax3.set_xticklabels(assets, fontsize=8, rotation=45, ha='right')
ax3.set_title('Portfolio Weights: MVO vs Regularized', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Weight', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')

# Plot 4: When to use what
ax4 = axes[1, 1]
ax4.axis('off')

guide = '''
REGULARIZATION IN FINANCE: WHEN TO USE

OLS (No regularization):
- Small number of features
- Large sample size (n >> p)
- All features known to be relevant
- Theory-driven model (e.g., CAPM)

RIDGE:
- Many correlated features
- All factors potentially relevant
- Want to shrink but keep all
- Example: Multi-factor risk models

LASSO:
- Many candidate features
- Want automatic selection
- Need interpretable sparse model
- Example: Stock return prediction

ELASTIC NET (Ridge + Lasso):
- Best of both worlds
- Handles correlated features
- Also does selection
- Example: Machine learning alphas

PRACTICAL TIPS:
1. Always use cross-validation
2. Time-series: Use walk-forward CV
3. Start with Ridge (safer)
4. Use Lasso if interpretability matters
5. Watch for data snooping
'''

ax4.text(0.02, 0.98, guide, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Practical Guidelines', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
