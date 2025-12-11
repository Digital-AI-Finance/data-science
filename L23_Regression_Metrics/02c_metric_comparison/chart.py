"""RMSE vs MAE Comparison"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))


# Two models with different error patterns
models = ['Consistent\nErrors', 'With\nOutliers']
rmse_vals = [5.2, 12.8]
mae_vals = [4.1, 6.3]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, rmse_vals, width, label='RMSE', color=MLBLUE, edgecolor='black')
bars2 = ax.bar(x + width/2, mae_vals, width, label='MAE', color=MLGREEN, edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_title('RMSE vs MAE: Outlier Sensitivity', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_ylabel('Error ($)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Annotate the gap
ax.annotate(f'Gap = {rmse_vals[1] - mae_vals[1]:.1f}\n(outlier effect)',
            xy=(1, rmse_vals[1]), xytext=(1.3, rmse_vals[1] + 1),
            fontsize=9, arrowprops=dict(arrowstyle='->', color=MLRED))


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
