"""Event Markers - Highlighting specific dates"""
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
MLORANGE = '#FF7F0E'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 100 + np.cumsum(np.random.randn(100) * 2)
ax.plot(dates, prices, color=MLBLUE, linewidth=2)

# Mark events
events = [(20, 'Earnings', MLGREEN), (45, 'Fed Meeting', MLRED), (75, 'Dividend', MLORANGE)]
for idx, label, color in events:
    ax.axvline(dates[idx], color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.scatter([dates[idx]], [prices[idx]], color=color, s=100, zorder=5)
    ax.annotate(label, xy=(dates[idx], prices[idx]),
                xytext=(5, 10), textcoords='offset points',
                fontsize=9, color=color, fontweight='bold')

ax.set_title('Event Markers', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
