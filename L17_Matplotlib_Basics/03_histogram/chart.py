"""Histograms - Distribution visualization"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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
fig.suptitle('Histograms with matplotlib', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic histogram
ax1 = axes[0, 0]
data = np.random.normal(100, 15, 1000)

ax1.hist(data, bins=30, color=MLBLUE, alpha=0.7, edgecolor='black')
ax1.axvline(np.mean(data), color=MLRED, linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.1f}')
ax1.axvline(np.median(data), color=MLGREEN, linestyle=':', linewidth=2, label=f'Median: {np.median(data):.1f}')

ax1.set_title('Basic Histogram', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Density (normalized) histogram with PDF
ax2 = axes[0, 1]
data = np.random.normal(0, 2, 1000)

ax2.hist(data, bins=40, density=True, color=MLGREEN, alpha=0.5, edgecolor='black', label='Data')

# Overlay fitted normal PDF
x = np.linspace(data.min(), data.max(), 100)
pdf = stats.norm.pdf(x, np.mean(data), np.std(data))
ax2.plot(x, pdf, color=MLRED, linewidth=2.5, label='Normal PDF')

ax2.set_title('Density Histogram with PDF', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Multiple histograms
ax3 = axes[1, 0]
low_vol = np.random.normal(0, 1, 500)
high_vol = np.random.normal(0, 2.5, 500)

ax3.hist(low_vol, bins=40, alpha=0.5, color=MLBLUE, edgecolor='black', label=f'Low Vol (std={np.std(low_vol):.2f})')
ax3.hist(high_vol, bins=40, alpha=0.5, color=MLRED, edgecolor='black', label=f'High Vol (std={np.std(high_vol):.2f})')

ax3.set_title('Comparing Distributions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Return (%)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Histogram with cumulative
ax4 = axes[1, 1]
returns = np.random.normal(0.05, 2, 500)

ax4.hist(returns, bins=40, color=MLORANGE, alpha=0.7, edgecolor='black', label='Frequency')

ax4_twin = ax4.twinx()
ax4_twin.hist(returns, bins=40, cumulative=True, density=True, histtype='step',
              color=MLPURPLE, linewidth=2.5, label='Cumulative')
ax4_twin.set_ylabel('Cumulative Probability', fontsize=10, color=MLPURPLE)

ax4.set_title('Histogram with Cumulative', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Return (%)', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.legend(loc='upper left', fontsize=9)
ax4_twin.legend(loc='center right', fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
