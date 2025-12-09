"""Line Plot - Basic line charts"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Line Plots with matplotlib', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic line plot
ax1 = axes[0, 0]
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax1.plot(x, y, color=MLBLUE, linewidth=2)
ax1.set_title('Basic: plt.plot(x, y)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Multiple lines with styling
ax2 = axes[0, 1]
dates = pd.date_range('2024-01-01', periods=100, freq='D')
price1 = 100 + np.cumsum(np.random.randn(100) * 2)
price2 = 100 + np.cumsum(np.random.randn(100) * 2)
price3 = 100 + np.cumsum(np.random.randn(100) * 2)

ax2.plot(dates, price1, color=MLBLUE, linewidth=2, linestyle='-', label='Stock A')
ax2.plot(dates, price2, color=MLGREEN, linewidth=2, linestyle='--', label='Stock B')
ax2.plot(dates, price3, color=MLRED, linewidth=2, linestyle=':', label='Stock C')

ax2.set_title('Multiple Lines with Styles', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Line with markers
ax3 = axes[1, 0]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
values = [100, 105, 102, 110, 108, 115]

ax3.plot(months, values, 'o-', color=MLORANGE, linewidth=2, markersize=10,
         markerfacecolor='white', markeredgewidth=2)

for i, v in enumerate(values):
    ax3.annotate(f'{v}', (i, v), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

ax3.set_title('Line with Markers and Labels', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Month', fontsize=10)
ax3.set_ylabel('Value', fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: Line plot with fill
ax4 = axes[1, 1]
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.sin(x) + 0.5

ax4.plot(x, y1, color=MLBLUE, linewidth=2, label='Lower bound')
ax4.plot(x, y2, color=MLGREEN, linewidth=2, label='Upper bound')
ax4.fill_between(x, y1, y2, alpha=0.3, color=MLPURPLE, label='Confidence band')

ax4.set_title('Fill Between Lines', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('X', fontsize=10)
ax4.set_ylabel('Y', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
