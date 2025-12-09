"""Seaborn Styling - Themes and aesthetics"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

# Generate sample data
x = np.linspace(0, 10, 50)
returns = np.random.normal(5, 2, 100)
prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: darkgrid style
with sns.axes_style("darkgrid"):
    ax1 = axes[0, 0]
    ax1.plot(prices, color=MLBLUE, linewidth=2)
    ax1.set_title('Style: darkgrid', fontsize=11, fontweight='bold', color=MLPURPLE)
    ax1.set_xlabel('Day', fontsize=10)
    ax1.set_ylabel('Price ($)', fontsize=10)
    ax1.tick_params(labelsize=9)

# Plot 2: whitegrid style
with sns.axes_style("whitegrid"):
    ax2 = axes[0, 1]
    ax2.hist(returns, bins=20, color=MLGREEN, alpha=0.7, edgecolor='black')
    ax2.set_title('Style: whitegrid', fontsize=11, fontweight='bold', color=MLPURPLE)
    ax2.set_xlabel('Return (%)', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.tick_params(labelsize=9)

# Plot 3: white style
with sns.axes_style("white"):
    ax3 = axes[1, 0]
    categories = ['Tech', 'Finance', 'Health', 'Energy']
    values = [15.2, 8.5, 12.1, 6.3]
    bars = ax3.bar(categories, values, color=[MLBLUE, MLGREEN, MLORANGE, MLRED])
    ax3.set_title('Style: white', fontsize=11, fontweight='bold', color=MLPURPLE)
    ax3.set_ylabel('Return (%)', fontsize=10)
    ax3.tick_params(labelsize=9)
    sns.despine(ax=ax3)  # Remove top and right spines

# Plot 4: ticks style
with sns.axes_style("ticks"):
    ax4 = axes[1, 1]
    scatter_x = np.random.uniform(5, 25, 50)
    scatter_y = 2 + 0.4 * scatter_x + np.random.normal(0, 2, 50)
    ax4.scatter(scatter_x, scatter_y, c=MLBLUE, s=60, alpha=0.7, edgecolors='black')
    ax4.set_title('Style: ticks', fontsize=11, fontweight='bold', color=MLPURPLE)
    ax4.set_xlabel('Risk (%)', fontsize=10)
    ax4.set_ylabel('Return (%)', fontsize=10)
    ax4.tick_params(labelsize=9)
    sns.despine(ax=ax4)

fig.suptitle('Seaborn Style Contexts', fontsize=14, fontweight='bold', color=MLPURPLE)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
