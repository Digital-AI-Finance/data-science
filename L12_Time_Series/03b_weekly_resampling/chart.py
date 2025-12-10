"""Weekly Resampling - Aggregating to lower frequency"""
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
MLBLUE = '#0066CC'

np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
df = pd.DataFrame({'Price': prices}, index=dates)

fig, ax = plt.subplots(figsize=(10, 6))

weekly = df.resample('W').agg({'Price': ['mean', 'min', 'max']})
weekly.columns = ['Mean', 'Min', 'Max']

ax.fill_between(weekly.index, weekly['Min'], weekly['Max'], alpha=0.3, color=MLBLUE, label='Range')
ax.plot(weekly.index, weekly['Mean'], color=MLBLUE, linewidth=2, label='Weekly Mean')
ax.scatter(weekly.index, weekly['Mean'], color=MLBLUE, s=30)

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title(f"Weekly: df.resample('W').mean() - {len(weekly)} obs", fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
