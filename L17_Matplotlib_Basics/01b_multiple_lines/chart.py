"""Multiple Lines - Different styles and legend"""
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
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

dates = pd.date_range('2024-01-01', periods=100, freq='D')
price1 = 100 + np.cumsum(np.random.randn(100) * 2)
price2 = 100 + np.cumsum(np.random.randn(100) * 2)
price3 = 100 + np.cumsum(np.random.randn(100) * 2)

ax.plot(dates, price1, color=MLBLUE, linewidth=2, linestyle='-', label='Stock A')
ax.plot(dates, price2, color=MLGREEN, linewidth=2, linestyle='--', label='Stock B')
ax.plot(dates, price3, color=MLRED, linewidth=2, linestyle=':', label='Stock C')

ax.set_title('Multiple Lines with Styles', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
