"""Correlation - NumPy for correlation analysis"""
import matplotlib.pyplot as plt
import numpy as np
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

# Generate correlated stock returns
np.random.seed(42)
n_days = 252

# Create correlated returns
market = np.random.normal(0.0004, 0.015, n_days)
aapl = 1.2 * market + np.random.normal(0, 0.008, n_days)  # High correlation with market
msft = 0.9 * market + np.random.normal(0, 0.010, n_days)
gold = -0.2 * market + np.random.normal(0.0002, 0.012, n_days)  # Low/negative correlation

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Correlation Analysis with NumPy', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Scatter - High correlation
ax1 = axes[0, 0]
corr_aapl = np.corrcoef(market, aapl)[0, 1]
ax1.scatter(market * 100, aapl * 100, alpha=0.5, color=MLBLUE, s=30)

# Add regression line
z = np.polyfit(market * 100, aapl * 100, 1)
p = np.poly1d(z)
ax1.plot(sorted(market * 100), p(sorted(market * 100)), color=MLRED, linewidth=2)

ax1.set_xlabel('Market Return (%)', fontsize=10)
ax1.set_ylabel('AAPL Return (%)', fontsize=10)
ax1.set_title(f'High Correlation: r = {corr_aapl:.3f}', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.grid(alpha=0.3)

# Plot 2: Scatter - Low correlation
ax2 = axes[0, 1]
corr_gold = np.corrcoef(market, gold)[0, 1]
ax2.scatter(market * 100, gold * 100, alpha=0.5, color=MLORANGE, s=30)

# Add regression line
z2 = np.polyfit(market * 100, gold * 100, 1)
p2 = np.poly1d(z2)
ax2.plot(sorted(market * 100), p2(sorted(market * 100)), color=MLRED, linewidth=2)

ax2.set_xlabel('Market Return (%)', fontsize=10)
ax2.set_ylabel('Gold Return (%)', fontsize=10)
ax2.set_title(f'Low/Negative Correlation: r = {corr_gold:.3f}', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3)

# Plot 3: Correlation matrix heatmap
ax3 = axes[1, 0]
assets = ['Market', 'AAPL', 'MSFT', 'Gold']
data = np.column_stack([market, aapl, msft, gold])
corr_matrix = np.corrcoef(data.T)

im = ax3.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)

ax3.set_xticks(range(len(assets)))
ax3.set_yticks(range(len(assets)))
ax3.set_xticklabels(assets)
ax3.set_yticklabels(assets)

# Add correlation values
for i in range(len(assets)):
    for j in range(len(assets)):
        text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=10,
                       color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

ax3.set_title('np.corrcoef() Correlation Matrix', fontsize=11, fontweight='bold', color=MLPURPLE)
plt.colorbar(im, ax=ax3, label='Correlation')

# Plot 4: Rolling correlation
ax4 = axes[1, 1]
window = 30
rolling_corr = []
for i in range(window, len(market)):
    corr = np.corrcoef(market[i-window:i], aapl[i-window:i])[0, 1]
    rolling_corr.append(corr)

ax4.plot(range(window, len(market)), rolling_corr, color=MLPURPLE, linewidth=1.5)
ax4.axhline(np.mean(rolling_corr), color=MLRED, linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(rolling_corr):.2f}')
ax4.fill_between(range(window, len(market)), rolling_corr,
                 np.mean(rolling_corr), alpha=0.3, color=MLBLUE)

ax4.set_xlabel('Trading Day', fontsize=10)
ax4.set_ylabel('30-Day Rolling Correlation', fontsize=10)
ax4.set_title('Rolling Correlation: AAPL vs Market', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=9)
ax4.set_ylim(0, 1)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
