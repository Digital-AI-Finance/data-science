"""Multidimensional Scatter - Size and color encoding"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

n = 50
market_cap = np.random.uniform(10, 500, n)
returns_pct = np.random.normal(5, 10, n)
volatility = np.random.uniform(10, 40, n)

scatter = ax.scatter(volatility, returns_pct, c=market_cap, s=market_cap/2,
                     cmap='viridis', alpha=0.7, edgecolors='black')
plt.colorbar(scatter, ax=ax, label='Market Cap ($B)')

ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_title('Multi-dimensional: Size = MarketCap, Color = MarketCap', fontsize=11,
             fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Volatility (%)', fontsize=10)
ax.set_ylabel('Return (%)', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
