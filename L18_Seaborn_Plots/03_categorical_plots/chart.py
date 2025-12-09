"""Categorical Plots - Visualizing categorical data with seaborn"""
import matplotlib.pyplot as plt
import seaborn as sns
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

# Create comprehensive finance dataset
sectors = ['Tech', 'Finance', 'Health', 'Energy', 'Consumer']
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

data = []
for sector in sectors:
    base_return = np.random.uniform(-2, 5)
    volatility = np.random.uniform(2, 6)
    for quarter in quarters:
        for _ in range(30):  # 30 companies per sector-quarter
            data.append({
                'Sector': sector,
                'Quarter': quarter,
                'Return': np.random.normal(base_return, volatility)
            })

df = pd.DataFrame(data)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Seaborn Categorical Plots', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: boxplot
ax1 = axes[0, 0]
sns.boxplot(data=df, x='Sector', y='Return', ax=ax1,
            palette=[MLBLUE, MLGREEN, MLORANGE, MLRED, MLPURPLE],
            linewidth=1.5)

ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_title('boxplot: Distribution by Category', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Sector', fontsize=10)
ax1.set_ylabel('Return (%)', fontsize=10)

# Plot 2: violinplot
ax2 = axes[0, 1]
sns.violinplot(data=df, x='Sector', y='Return', ax=ax2,
               palette=[MLBLUE, MLGREEN, MLORANGE, MLRED, MLPURPLE],
               inner='box', linewidth=1)

ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_title('violinplot: Distribution Shape + Box', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Sector', fontsize=10)
ax2.set_ylabel('Return (%)', fontsize=10)

# Plot 3: stripplot with swarmplot
ax3 = axes[1, 0]

# Subset data for clarity
df_subset = df.groupby(['Sector', 'Quarter']).head(10).reset_index(drop=True)

sns.stripplot(data=df_subset, x='Sector', y='Return', hue='Quarter', ax=ax3,
              palette=[MLBLUE, MLGREEN, MLORANGE, MLRED],
              dodge=True, alpha=0.7, size=5, jitter=0.2)

ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_title('stripplot: Individual Points by Group', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Sector', fontsize=10)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.legend(title='Quarter', fontsize=8, title_fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')

# Plot 4: barplot with error bars
ax4 = axes[1, 1]

sns.barplot(data=df, x='Sector', y='Return', hue='Quarter', ax=ax4,
            palette=[MLBLUE, MLGREEN, MLORANGE, MLRED],
            errorbar='sd', capsize=0.05, errwidth=1.5)

ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
ax4.set_title('barplot: Mean + Std Dev Error Bars', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Sector', fontsize=10)
ax4.set_ylabel('Mean Return (%)', fontsize=10)
ax4.legend(title='Quarter', fontsize=8, title_fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
