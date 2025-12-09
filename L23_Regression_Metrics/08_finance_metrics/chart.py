"""Finance Metrics - Domain-specific evaluation"""
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
fig.suptitle('Finance-Specific Model Evaluation Metrics', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate prediction data
n = 252  # Trading days
actual_returns = np.random.normal(0.05, 2, n)
predicted_returns = 0.3 * actual_returns + np.random.normal(0, 1.5, n)

# Plot 1: Information Coefficient (IC) - correlation
ax1 = axes[0, 0]

ax1.scatter(predicted_returns, actual_returns, c=MLBLUE, s=30, alpha=0.5, edgecolors='none')

# Fit line
z = np.polyfit(predicted_returns, actual_returns, 1)
x_line = np.linspace(predicted_returns.min(), predicted_returns.max(), 100)
ax1.plot(x_line, np.poly1d(z)(x_line), color=MLRED, linewidth=2.5)

# Calculate IC
ic = np.corrcoef(predicted_returns, actual_returns)[0, 1]

ax1.axhline(0, color='gray', linewidth=1, linestyle='--')
ax1.axvline(0, color='gray', linewidth=1, linestyle='--')

ax1.set_title('Information Coefficient (IC)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Predicted Return (%)', fontsize=10)
ax1.set_ylabel('Actual Return (%)', fontsize=10)
ax1.grid(alpha=0.3)

ax1.text(0.05, 0.95, f'IC = {ic:.3f}\n(Correlation between\npredicted and actual)',
         transform=ax1.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 2: IC over time (rolling)
ax2 = axes[0, 1]

# Calculate rolling IC
window = 20
rolling_ic = []
for i in range(window, n):
    ic_window = np.corrcoef(predicted_returns[i-window:i], actual_returns[i-window:i])[0, 1]
    rolling_ic.append(ic_window)

ax2.plot(range(window, n), rolling_ic, color=MLBLUE, linewidth=1.5)
ax2.fill_between(range(window, n), 0, rolling_ic,
                  where=[ic > 0 for ic in rolling_ic], alpha=0.3, color=MLGREEN)
ax2.fill_between(range(window, n), 0, rolling_ic,
                  where=[ic < 0 for ic in rolling_ic], alpha=0.3, color=MLRED)

ax2.axhline(0, color='gray', linewidth=1.5, linestyle='--')
ax2.axhline(np.mean(rolling_ic), color=MLORANGE, linewidth=2, linestyle='-.',
            label=f'Mean IC: {np.mean(rolling_ic):.3f}')

ax2.set_title('Rolling IC (20-day window)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Day', fontsize=10)
ax2.set_ylabel('IC', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Quintile analysis
ax3 = axes[1, 0]

# Sort predictions into quintiles and get actual returns
df = pd.DataFrame({'pred': predicted_returns, 'actual': actual_returns})
df['quintile'] = pd.qcut(df['pred'], 5, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])

quintile_returns = df.groupby('quintile', observed=True)['actual'].mean()

colors = [MLRED, MLORANGE, MLLAVENDER, MLBLUE, MLGREEN]
bars = ax3.bar(quintile_returns.index, quintile_returns.values, color=colors, edgecolor='black', linewidth=0.5)

ax3.axhline(0, color='gray', linewidth=1)

# Add trend line
ax3.plot(range(5), quintile_returns.values, color='black', linewidth=2, linestyle='--', marker='o')

ax3.set_title('Quintile Spread: Actual Returns by Predicted Rank', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Prediction Quintile', fontsize=10)
ax3.set_ylabel('Mean Actual Return (%)', fontsize=10)
ax3.grid(alpha=0.3, axis='y')

# Add spread annotation
spread = quintile_returns.iloc[-1] - quintile_returns.iloc[0]
ax3.text(2, quintile_returns.max() + 0.2, f'Q5-Q1 Spread: {spread:.2f}%',
         ha='center', fontsize=10, fontweight='bold', color=MLGREEN if spread > 0 else MLRED)

# Plot 4: Finance metrics summary
ax4 = axes[1, 1]
ax4.axis('off')

summary = f'''
FINANCE-SPECIFIC METRICS

1. INFORMATION COEFFICIENT (IC)
   - Correlation(predicted, actual)
   - Range: -1 to +1
   - IC > 0.05 is often valuable
   - Current: {ic:.3f}

2. IC INFORMATION RATIO (ICIR)
   - Mean(IC) / Std(IC)
   - Consistency of predictions
   - ICIR > 0.5 is good
   - Current: {np.mean(rolling_ic)/np.std(rolling_ic):.3f}

3. QUINTILE SPREAD
   - Return(Top) - Return(Bottom)
   - Tests if predictions rank correctly
   - Current: {spread:.2f}%

4. HIT RATE
   - % of correct direction predictions
   - > 50% indicates skill
   - Current: {np.mean((predicted_returns > 0) == (actual_returns > 0))*100:.1f}%

WHY NOT JUST USE R-SQUARED?
- In finance, R-sq is often < 5%
- But small R-sq can be very profitable!
- Focus on: direction, ranking, consistency
'''

ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Finance Metrics Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
