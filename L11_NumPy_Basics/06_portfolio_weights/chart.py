"""Portfolio Weights - Array operations for portfolio management"""
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

# Portfolio setup
np.random.seed(42)
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
returns = np.array([0.12, 0.15, 0.08, 0.18, 0.10])  # Annual returns
volatilities = np.array([0.25, 0.22, 0.20, 0.30, 0.35])  # Annual vol

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NumPy for Portfolio Analysis', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Portfolio weights
ax1 = axes[0, 0]
colors = [MLBLUE, MLORANGE, MLGREEN, MLRED, MLPURPLE]
wedges, texts, autotexts = ax1.pie(weights, labels=stocks, autopct='%1.0f%%',
                                   colors=colors, explode=[0.02]*5,
                                   textprops={'fontsize': 10})
ax1.set_title('Portfolio Weights\nweights = np.array([0.25, 0.20, ...])', fontsize=11,
              fontweight='bold', color=MLPURPLE)

# Verify weights sum to 1
ax1.text(0, -1.3, f'np.sum(weights) = {np.sum(weights):.2f}', ha='center',
         fontsize=10, family='monospace', color=MLGREEN)

# Plot 2: Weighted portfolio return
ax2 = axes[0, 1]
# Portfolio return = sum of weight * return for each stock
contrib = weights * returns * 100  # Contribution of each stock
portfolio_return = np.sum(weights * returns) * 100

bars = ax2.bar(stocks, contrib, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(portfolio_return / len(stocks), color='gray', linestyle='--', alpha=0.5)

ax2.set_ylabel('Return Contribution (%)', fontsize=10)
ax2.set_title(f'Weighted Return: np.sum(weights * returns) = {portfolio_return:.1f}%',
              fontsize=10, fontweight='bold', color=MLPURPLE)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, c in zip(bars, contrib):
    ax2.annotate(f'{c:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

# Plot 3: Risk-weighted analysis
ax3 = axes[1, 0]
risk_contrib = weights * volatilities * 100
portfolio_vol = np.sqrt(np.sum((weights * volatilities)**2)) * 100  # Simplified

bars3 = ax3.bar(stocks, risk_contrib, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(portfolio_vol / len(stocks), color=MLRED, linestyle='--', linewidth=2,
            label=f'Avg: {portfolio_vol/len(stocks):.1f}%')

ax3.set_ylabel('Risk Contribution (%)', fontsize=10)
ax3.set_title('Risk Contribution: weights * volatilities', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Rebalancing example
ax4 = axes[1, 1]
# After market moves, weights drift
market_moves = np.array([1.15, 1.08, 0.95, 1.20, 0.90])  # Price multipliers
new_values = weights * market_moves
drifted_weights = new_values / np.sum(new_values)

x = np.arange(len(stocks))
width = 0.35

bars_orig = ax4.bar(x - width/2, weights * 100, width, label='Target', color=MLBLUE, alpha=0.7)
bars_drift = ax4.bar(x + width/2, drifted_weights * 100, width, label='After Drift', color=MLORANGE, alpha=0.7)

ax4.set_ylabel('Weight (%)', fontsize=10)
ax4.set_title('Rebalancing: Weights Drift Over Time', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xticks(x)
ax4.set_xticklabels(stocks)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# Highlight largest drift
max_drift_idx = np.argmax(np.abs(weights - drifted_weights))
ax4.annotate(f'Largest drift', xy=(max_drift_idx + width/2, drifted_weights[max_drift_idx]*100),
             xytext=(max_drift_idx + 0.7, drifted_weights[max_drift_idx]*100 + 3),
             arrowprops=dict(arrowstyle='->', color=MLRED), fontsize=9, color=MLRED)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
