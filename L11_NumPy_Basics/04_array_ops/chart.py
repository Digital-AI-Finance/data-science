"""Array Operations - Common NumPy array operations"""
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

# Generate sample financial data
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 100)
prices = 100 * np.cumprod(1 + returns)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NumPy Array Operations for Finance', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Slicing and indexing
ax1 = axes[0, 0]
x = np.arange(len(prices))
ax1.plot(x, prices, color=MLPURPLE, linewidth=1.5, label='All prices')
ax1.plot(x[:20], prices[:20], color=MLGREEN, linewidth=3, label='First 20 (prices[:20])')
ax1.plot(x[-20:], prices[-20:], color=MLORANGE, linewidth=3, label='Last 20 (prices[-20:])')
ax1.scatter([50], [prices[50]], color=MLRED, s=100, zorder=5, label='Day 50 (prices[50])')

ax1.set_xlabel('Trading Day', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title('Array Slicing: prices[start:stop]', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Boolean masking
ax2 = axes[0, 1]
positive = returns > 0
negative = returns <= 0

ax2.bar(np.where(positive)[0], returns[positive] * 100, color=MLGREEN, alpha=0.7, label='Gains')
ax2.bar(np.where(negative)[0], returns[negative] * 100, color=MLRED, alpha=0.7, label='Losses')
ax2.axhline(0, color='black', linewidth=0.5)

ax2.set_xlabel('Trading Day', fontsize=10)
ax2.set_ylabel('Return (%)', fontsize=10)
ax2.set_title('Boolean Masking: returns[returns > 0]', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Add stats
pos_pct = np.sum(positive) / len(returns) * 100
ax2.text(0.98, 0.98, f'Positive days: {pos_pct:.1f}%', transform=ax2.transAxes,
         ha='right', va='top', fontsize=9, color=MLGREEN, fontweight='bold')

# Plot 3: Aggregation operations
ax3 = axes[1, 0]
operations = ['mean', 'std', 'min', 'max', 'sum']
values = [returns.mean() * 100, returns.std() * 100, returns.min() * 100,
          returns.max() * 100, returns.sum() * 100 / 10]  # Sum scaled for display
colors = [MLBLUE, MLORANGE, MLRED, MLGREEN, MLPURPLE]

bars = ax3.bar(operations, values, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(0, color='black', linewidth=0.5)

ax3.set_ylabel('Value (%)', fontsize=10)
ax3.set_title('Aggregation: np.mean(), np.std(), etc.', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, [returns.mean()*100, returns.std()*100, returns.min()*100, returns.max()*100, returns.sum()*100/10]):
    height = bar.get_height()
    ax3.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3 if height >= 0 else -10), textcoords="offset points",
                ha='center', fontsize=9, fontweight='bold')

# Plot 4: Reshaping
ax4 = axes[1, 1]
# Show monthly returns (reshape to 10 months x 10 days)
monthly = returns.reshape(10, 10)
monthly_avg = monthly.mean(axis=1) * 100

ax4.bar(range(1, 11), monthly_avg, color=MLBLUE, alpha=0.7, edgecolor='black')
ax4.axhline(returns.mean() * 100, color=MLRED, linestyle='--', linewidth=2,
            label=f'Overall mean: {returns.mean()*100:.2f}%')

ax4.set_xlabel('Month', fontsize=10)
ax4.set_ylabel('Avg Daily Return (%)', fontsize=10)
ax4.set_title('Reshaping: returns.reshape(10, 10)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
