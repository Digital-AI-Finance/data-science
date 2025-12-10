"""Monthly OHLC - Candlestick-style aggregation"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
df = pd.DataFrame({'Price': prices}, index=dates)

fig, ax = plt.subplots(figsize=(10, 6))

monthly = df.resample('M').agg({'Price': ['first', 'max', 'min', 'last']})
monthly.columns = ['Open', 'High', 'Low', 'Close']

for i, (date, row) in enumerate(monthly.iterrows()):
    color = MLGREEN if row['Close'] >= row['Open'] else MLRED
    ax.bar(date, abs(row['Close'] - row['Open']),
           bottom=min(row['Open'], row['Close']),
           width=15, color=color, alpha=0.7)
    ax.plot([date, date], [row['Low'], row['High']], color='black', linewidth=1)

ax.set_xlabel('Month', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title("Monthly OHLC: resample('M').agg(['first','max','min','last'])", fontsize=11,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
