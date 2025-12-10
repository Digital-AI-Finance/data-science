"""Beta Comparison - Bar chart of stock betas"""
import matplotlib.pyplot as plt
import numpy as np
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

fig, ax = plt.subplots(figsize=(10, 6))

# Stock betas (realistic examples)
stocks = ['JNJ\n(Healthcare)', 'KO\n(Consumer)', 'SPY\n(Index)', 'AAPL\n(Tech)', 'TSLA\n(EV)']
betas = [0.65, 0.75, 1.00, 1.25, 1.85]
colors = [MLGREEN, MLGREEN, MLBLUE, MLORANGE, MLRED]

bars = ax.bar(stocks, betas, color=colors, edgecolor='black', linewidth=0.5)

# Reference line at beta = 1
ax.axhline(1, color='gray', linestyle='--', linewidth=2, label='Market Beta = 1')

# Add value labels on bars
for bar, val in zip(bars, betas):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
            ha='center', fontsize=11, fontweight='bold')

# Add interpretation zones
ax.axhspan(0, 1, alpha=0.08, color=MLGREEN)
ax.axhspan(1, 2, alpha=0.08, color=MLRED)

ax.text(0.02, 0.45, 'Defensive\n(Less volatile)', transform=ax.transAxes, fontsize=9, color=MLGREEN, fontweight='bold')
ax.text(0.02, 0.75, 'Aggressive\n(More volatile)', transform=ax.transAxes, fontsize=9, color=MLRED, fontweight='bold')

ax.set_title('Beta Comparison Across Stocks', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_ylabel('Beta', fontsize=10)
ax.set_ylim(0, 2.1)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
