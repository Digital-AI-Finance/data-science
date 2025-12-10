"""Vertical Bar - Stock Prices"""
import matplotlib.pyplot as plt
import numpy as np
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

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
values = [185, 378, 145, 178, 325]
colors = [MLBLUE, MLGREEN, MLRED, MLORANGE, MLPURPLE]

bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'${height}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

ax.set_title('Vertical Bar: Stock Prices', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_ylabel('Price ($)', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
