"""Grouped Bar - Revenue vs Profit"""
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

fig, ax = plt.subplots(figsize=(10, 6))

quarters = ['Q1', 'Q2', 'Q3', 'Q4']
revenue = [100, 120, 115, 130]
profit = [20, 25, 22, 30]

x = np.arange(len(quarters))
width = 0.35

bars1 = ax.bar(x - width/2, revenue, width, color=MLBLUE, alpha=0.8, label='Revenue', edgecolor='black')
bars2 = ax.bar(x + width/2, profit, width, color=MLGREEN, alpha=0.8, label='Profit', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(quarters)
ax.set_title('Grouped Bar: Revenue vs Profit', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_ylabel('$ Millions', fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
