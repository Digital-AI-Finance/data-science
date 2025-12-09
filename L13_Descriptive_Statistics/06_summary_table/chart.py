"""Summary Statistics Table - df.describe() and beyond"""
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

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Summary Statistics with df.describe()', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE, transform=ax.transAxes)

# Create sample data
stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
data = {
    'Stock': stocks,
    'Count': [252, 252, 252, 252],
    'Mean': [0.08, 0.06, 0.05, 0.10],
    'Std': [1.8, 1.5, 2.1, 2.5],
    'Min': [-8.2, -6.5, -9.1, -11.5],
    '25%': [-1.0, -0.8, -1.2, -1.4],
    '50%': [0.05, 0.04, 0.03, 0.08],
    '75%': [1.1, 0.9, 1.3, 1.5],
    'Max': [7.5, 6.2, 8.8, 10.2],
}

# Table header
headers = ['Stock', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
col_widths = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
start_x = 0.1

# Draw header
y = 0.78
for i, (header, width) in enumerate(zip(headers, col_widths)):
    x = start_x + sum(col_widths[:i])
    ax.text(x, y, header, fontsize=11, fontweight='bold', color='white',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=MLPURPLE, edgecolor='none'))

# Draw data rows
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]
for row_idx, stock in enumerate(stocks):
    y = 0.68 - row_idx * 0.1
    row_color = colors[row_idx]

    for col_idx, header in enumerate(headers):
        x = start_x + sum(col_widths[:col_idx])
        value = data[header][row_idx]

        if header == 'Stock':
            ax.text(x, y, value, fontsize=10, fontweight='bold', color=row_color,
                    transform=ax.transAxes)
        else:
            if isinstance(value, float):
                text = f'{value:.2f}'
            else:
                text = str(value)
            ax.text(x, y, text, fontsize=10, transform=ax.transAxes)

# Additional statistics section
ax.text(0.5, 0.32, 'Additional Statistics (Beyond describe())', ha='center',
        fontsize=13, fontweight='bold', color=MLPURPLE, transform=ax.transAxes)

extra_stats = [
    ('Skewness', 'df.skew()', 'Asymmetry of distribution'),
    ('Kurtosis', 'df.kurtosis()', 'Tail heaviness'),
    ('Variance', 'df.var()', 'Squared standard deviation'),
    ('Mode', 'df.mode()', 'Most frequent value'),
    ('Range', 'df.max() - df.min()', 'Spread of data'),
    ('CV', 'df.std() / df.mean()', 'Coefficient of variation'),
]

y = 0.25
for i, (name, code, desc) in enumerate(extra_stats):
    col = i % 2
    row = i // 2
    x = 0.1 + col * 0.45
    y_pos = 0.25 - row * 0.08

    ax.text(x, y_pos, name + ':', fontsize=10, fontweight='bold', color=MLBLUE,
            transform=ax.transAxes)
    ax.text(x + 0.12, y_pos, code, fontsize=9, family='monospace',
            transform=ax.transAxes)
    ax.text(x + 0.12, y_pos - 0.025, desc, fontsize=8, color='gray',
            transform=ax.transAxes)

# Code example
ax.text(0.5, 0.02, "Usage: stats = df['returns'].describe()  |  full_stats = df.agg(['mean', 'std', 'skew', 'kurtosis'])",
        ha='center', fontsize=9, family='monospace', color=MLPURPLE, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
