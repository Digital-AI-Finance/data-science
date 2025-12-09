"""Central Tendency - Mean, Median, Mode"""
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
fig.suptitle('Measures of Central Tendency', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Symmetric distribution
ax1 = axes[0, 0]
data_sym = np.random.normal(100, 15, 1000)
ax1.hist(data_sym, bins=30, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')
mean_val = np.mean(data_sym)
median_val = np.median(data_sym)
ax1.axvline(mean_val, color=MLRED, linewidth=2.5, label=f'Mean: {mean_val:.1f}')
ax1.axvline(median_val, color=MLGREEN, linewidth=2.5, linestyle='--', label=f'Median: {median_val:.1f}')
ax1.set_title('Symmetric: Mean = Median', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Right-skewed (typical stock returns)
ax2 = axes[0, 1]
data_skew = np.random.exponential(50, 1000) + 20
ax2.hist(data_skew, bins=30, density=True, alpha=0.7, color=MLORANGE, edgecolor='black')
mean_val = np.mean(data_skew)
median_val = np.median(data_skew)
ax2.axvline(mean_val, color=MLRED, linewidth=2.5, label=f'Mean: {mean_val:.1f}')
ax2.axvline(median_val, color=MLGREEN, linewidth=2.5, linestyle='--', label=f'Median: {median_val:.1f}')
ax2.set_title('Right-Skewed: Mean > Median', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.annotate('Tail pulls mean right', xy=(mean_val + 30, 0.008), fontsize=9, color=MLRED)

# Plot 3: Stock returns comparison
ax3 = axes[1, 0]
returns = np.random.randn(252) * 2  # Daily returns
dates = pd.date_range('2024-01-01', periods=252, freq='B')
ax3.plot(dates, returns, color=MLBLUE, alpha=0.7, linewidth=1)
ax3.axhline(np.mean(returns), color=MLRED, linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
ax3.axhline(np.median(returns), color=MLGREEN, linewidth=2, linestyle='--', label=f'Median: {np.median(returns):.2f}%')
ax3.axhline(0, color='black', linewidth=1)
ax3.fill_between(dates, returns, 0, where=returns > 0, alpha=0.3, color=MLGREEN)
ax3.fill_between(dates, returns, 0, where=returns < 0, alpha=0.3, color=MLRED)
ax3.set_title('Stock Daily Returns', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.tick_params(axis='x', rotation=45)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Summary comparison
ax4 = axes[1, 1]
categories = ['Symmetric\n(Normal)', 'Right-Skewed\n(Exponential)', 'Left-Skewed']
means = [100, 70, 130]
medians = [100, 50, 150]
x = np.arange(len(categories))
width = 0.35
ax4.bar(x - width/2, means, width, color=MLRED, alpha=0.7, label='Mean', edgecolor='black')
ax4.bar(x + width/2, medians, width, color=MLGREEN, alpha=0.7, label='Median', edgecolor='black')
ax4.set_xticks(x)
ax4.set_xticklabels(categories, fontsize=9)
ax4.set_ylabel('Value', fontsize=10)
ax4.set_title('Mean vs Median by Distribution Shape', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# Add annotations
for i, (m, md) in enumerate(zip(means, medians)):
    ax4.annotate(f'{m}', xy=(i - width/2, m), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.annotate(f'{md}', xy=(i + width/2, md), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
