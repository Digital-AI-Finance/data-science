"""Finance GroupBy - Portfolio and risk analysis applications"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

# Create portfolio data
np.random.seed(42)
n_days = 252  # One year
dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

# Multiple stocks with different characteristics
stocks = {
    'AAPL': {'sector': 'Tech', 'return': 0.0008, 'vol': 0.02},
    'MSFT': {'sector': 'Tech', 'return': 0.0007, 'vol': 0.018},
    'JPM': {'sector': 'Finance', 'return': 0.0004, 'vol': 0.015},
    'GS': {'sector': 'Finance', 'return': 0.0003, 'vol': 0.020},
    'JNJ': {'sector': 'Healthcare', 'return': 0.0003, 'vol': 0.012},
    'PFE': {'sector': 'Healthcare', 'return': 0.0002, 'vol': 0.018},
}

# Generate returns
data = []
for date in dates:
    for stock, params in stocks.items():
        ret = np.random.normal(params['return'], params['vol'])
        data.append({
            'Date': date,
            'Stock': stock,
            'Sector': params['sector'],
            'Return': ret,
            'Month': date.month
        })

df = pd.DataFrame(data)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Finance Applications of GroupBy', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Cumulative returns by stock
ax1 = axes[0, 0]
colors_stock = {'AAPL': MLBLUE, 'MSFT': '#4A90D9', 'JPM': MLORANGE, 'GS': '#FFB366',
                'JNJ': MLGREEN, 'PFE': '#66CC66'}

for stock in stocks.keys():
    stock_data = df[df['Stock'] == stock].copy()
    stock_data['Cumulative'] = (1 + stock_data['Return']).cumprod() - 1
    ax1.plot(stock_data['Date'], stock_data['Cumulative'] * 100,
             label=stock, color=colors_stock[stock], linewidth=1.5, alpha=0.8)

ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Cumulative Return (%)', fontsize=10)
ax1.set_title('Cumulative Returns by Stock', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(ncol=2, fontsize=8)
ax1.grid(alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Sector average performance
ax2 = axes[0, 1]
sector_daily = df.groupby(['Date', 'Sector'])['Return'].mean().unstack()
sector_cum = (1 + sector_daily).cumprod() - 1

sector_colors = {'Tech': MLBLUE, 'Finance': MLORANGE, 'Healthcare': MLGREEN}
for sector in sector_cum.columns:
    ax2.plot(sector_cum.index, sector_cum[sector] * 100, label=sector,
             color=sector_colors[sector], linewidth=2)

ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Cumulative Return (%)', fontsize=10)
ax2.set_title("Sector Performance: groupby(['Date','Sector']).mean()", fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Monthly returns by sector
ax3 = axes[1, 0]
monthly_sector = df.groupby(['Month', 'Sector'])['Return'].sum().unstack() * 100

x = np.arange(12)
width = 0.25
for i, sector in enumerate(['Tech', 'Finance', 'Healthcare']):
    ax3.bar(x + i*width, monthly_sector[sector], width, label=sector,
            color=sector_colors[sector], alpha=0.7)

ax3.set_xlabel('Month', fontsize=10)
ax3.set_ylabel('Monthly Return (%)', fontsize=10)
ax3.set_title("Monthly Returns: groupby(['Month','Sector']).sum()", fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax3.set_xticks(x + width)
ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=8)
ax3.legend(fontsize=9)
ax3.axhline(0, color='black', linewidth=0.5)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Risk metrics by stock
ax4 = axes[1, 1]
risk_metrics = df.groupby('Stock').agg({
    'Return': ['mean', 'std']
}).round(4)
risk_metrics.columns = ['Mean', 'Std']
risk_metrics['Sharpe'] = (risk_metrics['Mean'] * 252) / (risk_metrics['Std'] * np.sqrt(252))
risk_metrics = risk_metrics.sort_values('Sharpe', ascending=True)

colors_bar = [colors_stock[s] for s in risk_metrics.index]
bars = ax4.barh(risk_metrics.index, risk_metrics['Sharpe'], color=colors_bar, alpha=0.7, edgecolor='black')

ax4.set_xlabel('Sharpe Ratio (Annualized)', fontsize=10)
ax4.set_title('Risk-Adjusted Returns by Stock', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.axvline(0, color='black', linewidth=0.5)
ax4.grid(axis='x', alpha=0.3)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax4.annotate(f'{width:.2f}', xy=(width, bar.get_y() + bar.get_height()/2),
                xytext=(3, 0), textcoords="offset points",
                ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
