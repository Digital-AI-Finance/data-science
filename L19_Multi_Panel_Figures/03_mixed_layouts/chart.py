"""Mixed Layouts - Different sized subplots"""
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

# Create mixed layout using subplot2grid
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Mixed Layouts with subplot2grid', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
days = 252
prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.015))
returns = np.diff(np.log(prices)) * 100
volume = np.random.randint(100, 500, days).astype(float)

# Large main plot - spans 2 columns
ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
ax_main.plot(prices, color=MLBLUE, linewidth=2)
ax_main.fill_between(range(len(prices)), prices.min(), prices, alpha=0.2, color=MLBLUE)
ax_main.set_title('Main Chart: subplot2grid((3,3), (0,0), colspan=2, rowspan=2)',
                  fontsize=10, fontweight='bold', color=MLPURPLE)
ax_main.set_xlabel('Day', fontsize=10)
ax_main.set_ylabel('Price ($)', fontsize=10)
ax_main.grid(alpha=0.3)

# Small right plot 1
ax_right1 = plt.subplot2grid((3, 3), (0, 2))
ax_right1.hist(returns, bins=20, color=MLGREEN, alpha=0.7, edgecolor='black', orientation='horizontal')
ax_right1.axhline(0, color=MLRED, linestyle='--', linewidth=1.5)
ax_right1.set_title('Return Dist.', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_right1.set_xlabel('Count', fontsize=9)
ax_right1.set_ylabel('Return (%)', fontsize=9)

# Small right plot 2
ax_right2 = plt.subplot2grid((3, 3), (1, 2))
sectors = ['Tech', 'Fin', 'Health']
weights = [0.45, 0.30, 0.25]
ax_right2.pie(weights, labels=sectors, colors=[MLBLUE, MLGREEN, MLORANGE],
              autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9})
ax_right2.set_title('Allocation', fontsize=10, fontweight='bold', color=MLPURPLE)

# Bottom full-width plot
ax_bottom = plt.subplot2grid((3, 3), (2, 0), colspan=3)
colors = [MLGREEN if prices[i] > prices[i-1] else MLRED for i in range(1, len(prices))]
colors = [MLGREEN] + colors
ax_bottom.bar(range(len(volume)), volume, color=colors, alpha=0.6, width=1)
ax_bottom.set_title('Volume: subplot2grid((3,3), (2,0), colspan=3)', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_bottom.set_xlabel('Day', fontsize=10)
ax_bottom.set_ylabel('Volume', fontsize=10)
ax_bottom.set_xlim(0, len(volume))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
