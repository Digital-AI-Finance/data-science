"""Day of Week Analysis - Extract time attributes"""
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
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
n = len(dates)

trend = np.linspace(100, 150, n)
seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, n))
noise = np.cumsum(np.random.randn(n) * 0.5)
prices = trend + seasonal + noise

df = pd.DataFrame({'Date': dates, 'Price': prices})
df.set_index('Date', inplace=True)
df['DayOfWeek'] = df.index.dayofweek

fig, ax = plt.subplots(figsize=(10, 6))

dow_avg = df.groupby('DayOfWeek')['Price'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

bars = ax.bar(days, dow_avg, color=[MLBLUE]*5 + [MLGREEN]*2, alpha=0.7, edgecolor='black')
ax.axhline(dow_avg.mean(), color=MLRED, linestyle='--', linewidth=2, label='Overall mean')

ax.set_xlabel('Day of Week', fontsize=10)
ax.set_ylabel('Avg Price ($)', fontsize=10)
ax.set_title('df.index.dayofweek - Extract Time Attributes', fontsize=12,
             fontweight='bold', color=MLPURPLE)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

ax.annotate('Weekend', xy=(5.5, dow_avg.iloc[5]), xytext=(5.5, dow_avg.iloc[5]+3),
            fontsize=10, ha='center', color=MLGREEN, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
