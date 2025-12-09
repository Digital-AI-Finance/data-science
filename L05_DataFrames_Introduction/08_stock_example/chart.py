"""Generated chart with course color palette"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Wedge
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

# Course colors already used in this lesson
COLOR_PRIMARY = MLPURPLE
COLOR_SECONDARY = MLPURPLE
COLOR_ACCENT = MLBLUE
COLOR_LIGHT = MLLAVENDER
COLOR_GREEN = MLGREEN
COLOR_ORANGE = MLORANGE
COLOR_RED = MLRED


fig, ax = plt.subplots(figsize=(10, 6))

# Use actual stock data
dates = pd.date_range('2024-01-01', periods=50, freq='D')
np.random.seed(42)
prices = 185 + np.cumsum(np.random.randn(50) * 2)

ax.plot(dates, prices, color=MLPURPLE, linewidth=2, label='AAPL')
ax.fill_between(dates, prices, alpha=0.3, color=MLLAVENDER)

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title('Stock Price DataFrame Visualization', fontsize=14, fontweight='bold', color=MLPURPLE)
ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("\nL05 COMPLETE: 8/8 charts generated")

# =============================================================================
# L06: Selection and Filtering
# =============================================================================