"""Subplots Grid - Basic multi-panel layouts"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Create 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('plt.subplots(rows, cols) - Basic Grid Layout', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate finance data
days = 252
prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.015))
returns = np.diff(np.log(prices)) * 100
volume = np.random.randint(100, 500, days)

# Plot 1: Line chart (0, 0)
ax1 = axes[0, 0]
ax1.plot(prices, color=MLBLUE, linewidth=2)
ax1.set_title('axes[0, 0]: Price Chart', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Day', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Histogram (0, 1)
ax2 = axes[0, 1]
ax2.hist(returns, bins=30, color=MLGREEN, alpha=0.7, edgecolor='black')
ax2.axvline(0, color=MLRED, linestyle='--', linewidth=2)
ax2.set_title('axes[0, 1]: Return Distribution', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Return (%)', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: Bar chart (1, 0)
ax3 = axes[1, 0]
monthly_returns = [2.1, -1.5, 3.2, 1.8, -0.5, 2.3, 1.1, -2.1, 0.8, 3.5, -0.2, 1.9]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = [MLGREEN if r > 0 else MLRED for r in monthly_returns]
ax3.bar(months, monthly_returns, color=colors, edgecolor='black', linewidth=0.5)
ax3.axhline(0, color='gray', linewidth=1)
ax3.set_title('axes[1, 0]: Monthly Returns', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Month', fontsize=10)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(alpha=0.3, axis='y')

# Plot 4: Scatter (1, 1)
ax4 = axes[1, 1]
risk = np.random.uniform(10, 30, 30)
exp_return = 2 + 0.3 * risk + np.random.normal(0, 2, 30)
ax4.scatter(risk, exp_return, c=MLORANGE, s=80, alpha=0.7, edgecolors='black')
ax4.set_title('axes[1, 1]: Risk-Return', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Risk (%)', fontsize=10)
ax4.set_ylabel('Expected Return (%)', fontsize=10)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
