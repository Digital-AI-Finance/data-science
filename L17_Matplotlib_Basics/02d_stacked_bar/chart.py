"""Stacked Bar - Portfolio Allocation"""
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
MLORANGE = '#FF7F0E'

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Fund A', 'Fund B', 'Fund C', 'Fund D']
stocks = [60, 50, 70, 40]
bonds = [30, 35, 20, 45]
cash = [10, 15, 10, 15]

ax.bar(categories, stocks, color=MLBLUE, alpha=0.8, label='Stocks', edgecolor='black')
ax.bar(categories, bonds, bottom=stocks, color=MLGREEN, alpha=0.8, label='Bonds', edgecolor='black')
ax.bar(categories, cash, bottom=np.array(stocks)+np.array(bonds), color=MLORANGE, alpha=0.8, label='Cash', edgecolor='black')

ax.set_title('Stacked Bar: Portfolio Allocation', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_ylabel('Allocation (%)', fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
