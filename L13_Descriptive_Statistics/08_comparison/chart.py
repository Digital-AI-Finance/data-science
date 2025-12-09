"""Statistics Comparison - Comparing multiple assets"""
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
fig.suptitle('Comparing Statistics Across Assets', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data for multiple assets
n = 252
assets = {
    'US Stocks': np.random.normal(0.05, 1.5, n),
    'Bonds': np.random.normal(0.02, 0.4, n),
    'Gold': np.random.normal(0.03, 1.0, n),
    'Crypto': np.random.normal(0.15, 4.0, n),
}
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]

# Plot 1: Return vs Risk (scatter)
ax1 = axes[0, 0]
for (name, returns), color in zip(assets.items(), colors):
    mean_ret = np.mean(returns) * 252  # Annualized
    std_ret = np.std(returns) * np.sqrt(252)  # Annualized
    ax1.scatter(std_ret, mean_ret, s=200, color=color, edgecolors='black', linewidth=2, label=name)
    ax1.annotate(name, xy=(std_ret + 1, mean_ret + 1), fontsize=9)

ax1.set_title('Risk-Return Profile', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Annualized Volatility (%)', fontsize=10)
ax1.set_ylabel('Annualized Return (%)', fontsize=10)
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8, loc='lower right')

# Plot 2: Boxplot comparison
ax2 = axes[0, 1]
bp = ax2.boxplot([assets[a] for a in assets], patch_artist=True, labels=assets.keys())
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title('Daily Return Distributions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Daily Return (%)', fontsize=10)
ax2.tick_params(axis='x', rotation=15)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Radar/Spider chart approximation with bar
ax3 = axes[1, 0]
metrics = ['Return', 'Volatility', 'Sharpe', 'Skew', 'Kurtosis']

# Normalize metrics for comparison
data_norm = {}
for name, returns in assets.items():
    mean_ret = np.mean(returns) * 252
    std_ret = np.std(returns) * np.sqrt(252)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    skew = pd.Series(returns).skew()
    kurt = pd.Series(returns).kurtosis()
    data_norm[name] = [mean_ret, std_ret, sharpe, skew, kurt]

x = np.arange(len(metrics))
width = 0.2
for i, ((name, values), color) in enumerate(zip(data_norm.items(), colors)):
    # Normalize to 0-1 for visualization
    ax3.bar(x + i * width, values, width, color=color, alpha=0.7, label=name, edgecolor='black')

ax3.set_xticks(x + width * 1.5)
ax3.set_xticklabels(metrics, fontsize=9)
ax3.set_title('Metrics Comparison (Raw Values)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(0, color='black', linewidth=1)

# Plot 4: Cumulative returns
ax4 = axes[1, 1]
for (name, returns), color in zip(assets.items(), colors):
    cumulative = (1 + returns/100).cumprod()
    ax4.plot(cumulative, color=color, linewidth=2, label=name)

ax4.axhline(1, color='black', linestyle='--', linewidth=1)
ax4.set_title('Cumulative Returns (Starting at $1)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Trading Day', fontsize=10)
ax4.set_ylabel('Portfolio Value ($)', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Add final values
for (name, returns), color in zip(assets.items(), colors):
    cumulative = (1 + returns/100).cumprod()
    ax4.annotate(f'${cumulative[-1]:.2f}', xy=(n-1, cumulative[-1]),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=8, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
