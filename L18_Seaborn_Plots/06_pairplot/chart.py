"""Pairplot - Multi-variable relationship exploration"""
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

# Create comprehensive stock data for pairplot
n_stocks = 100
df = pd.DataFrame({
    'Return': np.random.normal(8, 12, n_stocks),
    'Volatility': np.abs(np.random.normal(20, 8, n_stocks)),
    'Beta': np.random.uniform(0.5, 2.0, n_stocks),
    'PE_Ratio': np.random.uniform(10, 40, n_stocks),
    'Sector': np.random.choice(['Tech', 'Finance', 'Health'], n_stocks)
})

# Add some realistic correlations
df['Return'] = df['Return'] + 0.3 * df['Beta'] * 5  # Higher beta = higher expected return
df['Volatility'] = df['Volatility'] + df['Beta'] * 8  # Higher beta = higher volatility

# Create pairplot
g = sns.pairplot(df, hue='Sector',
                 palette={'Tech': MLBLUE, 'Finance': MLGREEN, 'Health': MLORANGE},
                 diag_kind='kde',
                 plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'white', 'linewidth': 0.5},
                 diag_kws={'linewidth': 2},
                 height=2.5, aspect=1)

g.figure.suptitle('Seaborn pairplot: Multi-Variable Exploration',
                 fontsize=14, fontweight='bold', color=MLPURPLE, y=1.02)

# Adjust legend
g._legend.set_title('Sector')
g._legend.get_title().set_fontsize(10)
for text in g._legend.get_texts():
    text.set_fontsize(9)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
