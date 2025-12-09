"""Bar Charts - Categorical data visualization"""
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

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bar Charts with matplotlib', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic vertical bar
ax1 = axes[0, 0]
categories = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
values = [185, 378, 145, 178, 325]
colors = [MLBLUE, MLGREEN, MLRED, MLORANGE, MLPURPLE]

bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'${height}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

ax1.set_title('Vertical Bar: Stock Prices', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Horizontal bar
ax2 = axes[0, 1]
sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy']
returns = [12.5, 8.3, 6.7, 5.2, -2.1]
colors = [MLGREEN if r > 0 else MLRED for r in returns]

bars = ax2.barh(sectors, returns, color=colors, alpha=0.8, edgecolor='black')
ax2.axvline(0, color='black', linewidth=1)

for bar, val in zip(bars, returns):
    x_pos = val + 0.5 if val > 0 else val - 1
    ax2.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
             va='center', fontsize=9, fontweight='bold')

ax2.set_title('Horizontal Bar: Sector Returns', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Return (%)', fontsize=10)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Grouped bar
ax3 = axes[1, 0]
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
revenue = [100, 120, 115, 130]
profit = [20, 25, 22, 30]

x = np.arange(len(quarters))
width = 0.35

bars1 = ax3.bar(x - width/2, revenue, width, color=MLBLUE, alpha=0.8, label='Revenue', edgecolor='black')
bars2 = ax3.bar(x + width/2, profit, width, color=MLGREEN, alpha=0.8, label='Profit', edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(quarters)
ax3.set_title('Grouped Bar: Revenue vs Profit', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('$ Millions', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Stacked bar
ax4 = axes[1, 1]
categories = ['Fund A', 'Fund B', 'Fund C', 'Fund D']
stocks = [60, 50, 70, 40]
bonds = [30, 35, 20, 45]
cash = [10, 15, 10, 15]

ax4.bar(categories, stocks, color=MLBLUE, alpha=0.8, label='Stocks', edgecolor='black')
ax4.bar(categories, bonds, bottom=stocks, color=MLGREEN, alpha=0.8, label='Bonds', edgecolor='black')
ax4.bar(categories, cash, bottom=np.array(stocks)+np.array(bonds), color=MLORANGE, alpha=0.8, label='Cash', edgecolor='black')

ax4.set_title('Stacked Bar: Portfolio Allocation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_ylabel('Allocation (%)', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
