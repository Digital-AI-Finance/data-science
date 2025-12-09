"""Fama-French Factors - The classic multi-factor model"""
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
fig.suptitle('Fama-French Factor Model', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Factor definitions
ax1 = axes[0, 0]
ax1.axis('off')

definitions = '''
FAMA-FRENCH THREE-FACTOR MODEL (1993)

1. MKT (Market)
   - Market return minus risk-free rate
   - Rm - Rf
   - Captures broad market exposure

2. SMB (Small Minus Big)
   - Small cap return minus large cap return
   - Size effect: small stocks outperform
   - Based on market capitalization

3. HML (High Minus Low)
   - Value stocks minus growth stocks
   - Value effect: high B/M outperform
   - Based on Book-to-Market ratio

FAMA-FRENCH FIVE-FACTOR MODEL (2015)

4. RMW (Robust Minus Weak)
   - Profitable firms minus unprofitable
   - Profitability effect

5. CMA (Conservative Minus Aggressive)
   - Low investment minus high investment
   - Investment effect
'''

ax1.text(0.02, 0.98, definitions, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Factor Definitions', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Historical factor returns
ax2 = axes[0, 1]

# Simulated monthly factor returns (like Ken French data)
months = 120  # 10 years
dates = pd.date_range('2014-01-01', periods=months, freq='M')

mkt = np.random.normal(0.8, 4.5, months)
smb = np.random.normal(0.15, 3, months)
hml = np.random.normal(0.25, 3.5, months)

cumulative_mkt = np.cumprod(1 + mkt/100) - 1
cumulative_smb = np.cumprod(1 + smb/100) - 1
cumulative_hml = np.cumprod(1 + hml/100) - 1

ax2.plot(dates, cumulative_mkt * 100, color=MLBLUE, linewidth=2, label='MKT')
ax2.plot(dates, cumulative_smb * 100, color=MLGREEN, linewidth=2, label='SMB')
ax2.plot(dates, cumulative_hml * 100, color=MLORANGE, linewidth=2, label='HML')

ax2.axhline(0, color='gray', linewidth=1, linestyle='--')
ax2.set_title('Cumulative Factor Returns (10 Years)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Cumulative Return (%)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Factor statistics table
ax3 = axes[1, 0]
ax3.axis('off')

# Calculate statistics
factor_stats = {
    'MKT': {'mean': np.mean(mkt), 'std': np.std(mkt), 'sharpe': np.mean(mkt)/np.std(mkt)*np.sqrt(12)},
    'SMB': {'mean': np.mean(smb), 'std': np.std(smb), 'sharpe': np.mean(smb)/np.std(smb)*np.sqrt(12)},
    'HML': {'mean': np.mean(hml), 'std': np.std(hml), 'sharpe': np.mean(hml)/np.std(hml)*np.sqrt(12)}
}

stats_text = f'''
FACTOR STATISTICS (Monthly)

Factor    Mean      Std       Sharpe (Ann.)
---------------------------------------------
MKT      {factor_stats['MKT']['mean']:6.2f}%   {factor_stats['MKT']['std']:5.2f}%    {factor_stats['MKT']['sharpe']:.2f}
SMB      {factor_stats['SMB']['mean']:6.2f}%   {factor_stats['SMB']['std']:5.2f}%    {factor_stats['SMB']['sharpe']:.2f}
HML      {factor_stats['HML']['mean']:6.2f}%   {factor_stats['HML']['std']:5.2f}%    {factor_stats['HML']['sharpe']:.2f}

CORRELATION MATRIX

         MKT     SMB     HML
MKT     1.00   {np.corrcoef(mkt, smb)[0,1]:.2f}   {np.corrcoef(mkt, hml)[0,1]:.2f}
SMB     {np.corrcoef(smb, mkt)[0,1]:.2f}   1.00   {np.corrcoef(smb, hml)[0,1]:.2f}
HML     {np.corrcoef(hml, mkt)[0,1]:.2f}   {np.corrcoef(hml, smb)[0,1]:.2f}   1.00

Key: Low correlations = good diversification
'''

ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Factor Statistics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: sklearn implementation
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Loading Fama-French Data with pandas

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load Fama-French factors (from Ken French website)
# Format: Date, Mkt-RF, SMB, HML, RF (in %)
ff_data = pd.read_csv('F-F_Research_Data_Factors.csv',
                       skiprows=3)
ff_data['Date'] = pd.to_datetime(ff_data['Date'],
                                  format='%Y%m')

# Load stock returns
stock_returns = pd.read_csv('stock_returns.csv')

# Merge and prepare for regression
merged = pd.merge(stock_returns, ff_data, on='Date')
merged['excess_return'] = merged['return'] - merged['RF']

# Fit FF3 model
X = merged[['Mkt-RF', 'SMB', 'HML']]
y = merged['excess_return']

model = LinearRegression()
model.fit(X, y)

print(f"Alpha: {model.intercept_:.4f}")
print(f"MKT beta: {model.coef_[0]:.3f}")
print(f"SMB beta: {model.coef_[1]:.3f}")
print(f"HML beta: {model.coef_[2]:.3f}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
