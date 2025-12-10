"""Daily Data - Original time series frequency"""
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
ax.plot(df.index, df['Price'], color=MLBLUE, linewidth=1, alpha=0.7)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title(f'Daily Data: {len(df)} observations', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
