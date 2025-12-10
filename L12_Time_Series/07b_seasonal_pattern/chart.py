"""Seasonal Pattern - Monthly average after detrending"""
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

np.random.seed(42)
n = 365 * 2
dates = pd.date_range('2023-01-01', periods=n, freq='D')
trend = np.linspace(100, 180, n)
seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.randn(n) * 3
series = trend + seasonal + noise

df = pd.DataFrame({'Value': series}, index=dates)
detrended = df['Value'] - trend

monthly = df.copy()
monthly['Month'] = monthly.index.month
monthly['Detrended'] = detrended
monthly_avg = monthly.groupby('Month')['Detrended'].mean()

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(monthly_avg.index, monthly_avg, color=MLPURPLE, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linewidth=1)

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels, fontsize=9)
ax.set_xlabel('Month', fontsize=10)
ax.set_ylabel('Seasonal Effect', fontsize=10)
ax.set_title('Seasonal Pattern (Monthly Average)', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.grid(axis='y', alpha=0.3)

max_month = monthly_avg.idxmax()
ax.annotate(f'Peak: {month_labels[max_month-1]}',
            xy=(max_month, monthly_avg[max_month]),
            xytext=(max_month, monthly_avg[max_month] + 3),
            ha='center', fontsize=10, color=MLGREEN, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
