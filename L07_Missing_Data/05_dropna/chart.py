"""Dropna Effects - Visualization of data loss when dropping missing values"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Create synthetic dataset with various missing patterns
np.random.seed(42)
n = 100
data = {
    'Stock_A': np.random.randn(n) * 10 + 100,
    'Stock_B': np.random.randn(n) * 15 + 150,
    'Stock_C': np.random.randn(n) * 8 + 80,
    'Volume': np.random.randint(1000, 10000, n).astype(float),
    'Sentiment': np.random.uniform(-1, 1, n)
}
df = pd.DataFrame(data)

# Introduce missing values with different patterns
df.loc[np.random.choice(n, 10, replace=False), 'Stock_A'] = np.nan
df.loc[np.random.choice(n, 20, replace=False), 'Stock_B'] = np.nan
df.loc[np.random.choice(n, 5, replace=False), 'Stock_C'] = np.nan
df.loc[np.random.choice(n, 15, replace=False), 'Volume'] = np.nan
df.loc[np.random.choice(n, 25, replace=False), 'Sentiment'] = np.nan

# Calculate data remaining after different dropna strategies
original_rows = len(df)
dropna_any = len(df.dropna(how='any'))
dropna_all = len(df.dropna(how='all'))
dropna_thresh_3 = len(df.dropna(thresh=3))
dropna_thresh_4 = len(df.dropna(thresh=4))
dropna_subset_ab = len(df.dropna(subset=['Stock_A', 'Stock_B']))

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Bar chart of rows remaining
ax1 = axes[0]
strategies = ['Original', 'dropna()\nhow="any"', 'dropna()\nhow="all"',
              'dropna()\nthresh=3', 'dropna()\nthresh=4', 'dropna()\nsubset=[A,B]']
rows_remaining = [original_rows, dropna_any, dropna_all, dropna_thresh_3, dropna_thresh_4, dropna_subset_ab]
colors = [MLPURPLE, MLRED, MLGREEN, MLORANGE, MLBLUE, MLLAVENDER]

bars = ax1.bar(strategies, rows_remaining, color=colors, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Rows Remaining', fontsize=11)
ax1.set_title('Data Retention by dropna() Strategy', fontsize=12, fontweight='bold', color=MLPURPLE)
ax1.axhline(y=original_rows, color=MLPURPLE, linestyle='--', alpha=0.5, label='Original')

# Add percentage labels
for bar, rows in zip(bars, rows_remaining):
    pct = rows / original_rows * 100
    ax1.annotate(f'{rows}\n({pct:.0f}%)',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylim(0, 120)
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=0)

# Right: Heatmap showing missing values per column
ax2 = axes[1]
missing_pct = df.isnull().sum() / len(df) * 100
columns = list(df.columns)
y_pos = np.arange(len(columns))

bars2 = ax2.barh(y_pos, missing_pct, color=MLRED, alpha=0.7, edgecolor='black')
ax2.barh(y_pos, 100 - missing_pct, left=missing_pct, color=MLGREEN, alpha=0.7, edgecolor='black')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(columns)
ax2.set_xlabel('Percentage', fontsize=11)
ax2.set_title('Missing Values by Column', fontsize=12, fontweight='bold', color=MLPURPLE)
ax2.set_xlim(0, 100)

# Add labels
for i, (miss, pres) in enumerate(zip(missing_pct, 100 - missing_pct)):
    ax2.text(miss/2, i, f'{miss:.0f}%', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')
    ax2.text(miss + pres/2, i, f'{pres:.0f}%', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MLRED, alpha=0.7, label='Missing'),
                   Patch(facecolor=MLGREEN, alpha=0.7, label='Present')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

fig.suptitle('Impact of dropna() on Financial Dataset', fontsize=14,
             fontweight='bold', color=MLPURPLE, y=1.02)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
