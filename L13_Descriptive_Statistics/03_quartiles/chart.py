"""Quartiles and IQR - Boxplot fundamentals"""
import matplotlib.pyplot as plt
import numpy as np
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
fig.suptitle('Quartiles and Interquartile Range (IQR)', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Quartile explanation with boxplot anatomy
ax1 = axes[0, 0]
data = np.concatenate([np.random.normal(100, 15, 95), [50, 160]])  # With outliers
bp = ax1.boxplot(data, vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor(MLBLUE)
bp['boxes'][0].set_alpha(0.7)
bp['medians'][0].set_color(MLRED)
bp['medians'][0].set_linewidth(2)

# Annotate quartiles
q1, q2, q3 = np.percentile(data, [25, 50, 75])
ax1.annotate(f'Q1 = {q1:.1f}', xy=(1.1, q1), fontsize=9, color=MLGREEN)
ax1.annotate(f'Q2 (Median) = {q2:.1f}', xy=(1.1, q2), fontsize=9, color=MLRED, fontweight='bold')
ax1.annotate(f'Q3 = {q3:.1f}', xy=(1.1, q3), fontsize=9, color=MLORANGE)
ax1.annotate(f'IQR = {q3-q1:.1f}', xy=(0.6, (q1+q3)/2), fontsize=10, color=MLPURPLE, fontweight='bold')

ax1.set_title('Boxplot Anatomy', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Value', fontsize=10)
ax1.set_xticklabels(['Data'], fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Multiple stocks comparison
ax2 = axes[0, 1]
stocks = {
    'AAPL': np.random.normal(2, 3, 252),
    'MSFT': np.random.normal(1.5, 2.5, 252),
    'GOOG': np.random.normal(1, 4, 252),
    'META': np.random.normal(0.5, 5, 252),
}
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]
bp = ax2.boxplot([stocks[s] for s in stocks], patch_artist=True, labels=stocks.keys())
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title('Daily Returns Distribution by Stock', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Stock', fontsize=10)
ax2.set_ylabel('Daily Return (%)', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Percentiles visualization
ax3 = axes[1, 0]
data = np.random.normal(100, 20, 1000)
ax3.hist(data, bins=40, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')

percentiles = [5, 25, 50, 75, 95]
colors = [MLRED, MLORANGE, MLPURPLE, MLORANGE, MLRED]
labels = ['5th', 'Q1 (25th)', 'Median', 'Q3 (75th)', '95th']

for p, c, l in zip(percentiles, colors, labels):
    val = np.percentile(data, p)
    ax3.axvline(val, color=c, linewidth=2, linestyle='--', label=f'{l}: {val:.1f}')

ax3.set_title('Key Percentiles', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Value', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(alpha=0.3)

# Plot 4: Outlier detection with IQR
ax4 = axes[1, 1]
data = np.concatenate([np.random.normal(100, 10, 45), [30, 35, 170, 180, 190]])  # Add outliers
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = (data < lower_bound) | (data > upper_bound)
normal = ~outliers

ax4.scatter(range(len(data)), data, c=[MLRED if o else MLBLUE for o in outliers],
            s=50, alpha=0.7, edgecolors='black')

ax4.axhline(q1, color=MLORANGE, linestyle='--', linewidth=1.5, label=f'Q1: {q1:.1f}')
ax4.axhline(q3, color=MLORANGE, linestyle='--', linewidth=1.5, label=f'Q3: {q3:.1f}')
ax4.axhline(lower_bound, color=MLRED, linestyle=':', linewidth=2, label=f'Lower fence: {lower_bound:.1f}')
ax4.axhline(upper_bound, color=MLRED, linestyle=':', linewidth=2, label=f'Upper fence: {upper_bound:.1f}')

ax4.fill_between(range(len(data)), lower_bound, upper_bound, alpha=0.1, color=MLGREEN)

ax4.set_title(f'Outlier Detection: {sum(outliers)} outliers found', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Observation', fontsize=10)
ax4.set_ylabel('Value', fontsize=10)
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
