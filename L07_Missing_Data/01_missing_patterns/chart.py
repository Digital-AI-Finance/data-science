"""Missing Data Patterns - Heatmap visualization of missing values in stock data"""
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

# Generate synthetic stock data with missing values
np.random.seed(42)
n_rows = 50
dates = pd.date_range('2024-01-01', periods=n_rows, freq='D')

# Create data with different missing patterns
data = {
    'Date': dates,
    'AAPL': np.random.randn(n_rows).cumsum() + 150,
    'MSFT': np.random.randn(n_rows).cumsum() + 350,
    'GOOGL': np.random.randn(n_rows).cumsum() + 140,
    'Volume': np.random.randint(1000000, 5000000, n_rows).astype(float),
    'PE_Ratio': np.random.uniform(15, 35, n_rows)
}

df = pd.DataFrame(data)

# Introduce missing patterns
# MCAR - Missing Completely At Random (AAPL)
mcar_idx = np.random.choice(n_rows, 8, replace=False)
df.loc[mcar_idx, 'AAPL'] = np.nan

# MAR - Missing At Random (Volume missing when MSFT drops)
mar_idx = df[df['MSFT'] < df['MSFT'].quantile(0.3)].index[:6]
df.loc[mar_idx, 'Volume'] = np.nan

# MNAR - Missing Not At Random (PE_Ratio missing when high)
mnar_idx = df[df['PE_Ratio'] > 30].index
df.loc[mnar_idx, 'PE_Ratio'] = np.nan

# Consecutive missing (GOOGL)
df.loc[20:25, 'GOOGL'] = np.nan

# Create missing pattern heatmap
fig, ax = plt.subplots(figsize=(12, 7))

# Create binary missing matrix
cols_to_show = ['AAPL', 'MSFT', 'GOOGL', 'Volume', 'PE_Ratio']
missing_matrix = df[cols_to_show].isnull().astype(int).values

# Plot heatmap
cmap = plt.cm.colors.ListedColormap([MLLAVENDER, MLRED])
im = ax.imshow(missing_matrix.T, aspect='auto', cmap=cmap, interpolation='nearest')

# Labels
ax.set_yticks(range(len(cols_to_show)))
ax.set_yticklabels(cols_to_show)
ax.set_xlabel('Row Index (Trading Days)', fontsize=11)
ax.set_ylabel('Variables', fontsize=11)
ax.set_title('Missing Data Patterns in Stock Dataset', fontsize=14, fontweight='bold', color=MLPURPLE)

# Add pattern annotations
ax.annotate('MCAR\n(Random)', xy=(4, 0), xytext=(4, -0.8),
            fontsize=9, ha='center', color=MLPURPLE,
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=1.5))

ax.annotate('Consecutive\nMissing', xy=(22, 2), xytext=(35, 2),
            fontsize=9, ha='center', color=MLPURPLE,
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=1.5))

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MLLAVENDER, label='Present'),
                   Patch(facecolor=MLRED, label='Missing')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Summary stats
missing_pct = df[cols_to_show].isnull().sum() / len(df) * 100
summary_text = f"Missing %: AAPL={missing_pct['AAPL']:.0f}%, GOOGL={missing_pct['GOOGL']:.0f}%, Volume={missing_pct['Volume']:.0f}%, PE={missing_pct['PE_Ratio']:.0f}%"
ax.text(0.5, -0.12, summary_text, transform=ax.transAxes, fontsize=9,
        ha='center', color=MLPURPLE, style='italic')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
