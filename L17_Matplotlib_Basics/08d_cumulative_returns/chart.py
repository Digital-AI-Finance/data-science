"""Cumulative Returns - Strategy comparison"""
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
MLORANGE = '#FF7F0E'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

dates = pd.date_range('2023-01-01', periods=252, freq='B')

strategies = {
    'Portfolio': np.random.normal(0.0004, 0.01, 252),
    'Benchmark': np.random.normal(0.0003, 0.012, 252),
    'Risk-Free': np.full(252, 0.00015),
}

for name, returns in strategies.items():
    cumulative = (1 + returns).cumprod()
    color = MLBLUE if name == 'Portfolio' else (MLORANGE if name == 'Benchmark' else MLGREEN)
    style = '-' if name == 'Portfolio' else ('--' if name == 'Benchmark' else ':')
    ax.plot(dates, cumulative, color=color, linewidth=2, linestyle=style,
            label=f'{name}: {(cumulative[-1]-1)*100:.1f}%')

ax.axhline(1, color='gray', linestyle='-', linewidth=1)
ax.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Cumulative Return', fontsize=10)
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
