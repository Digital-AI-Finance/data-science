"""Dispersion - Variance and Standard Deviation"""
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
fig.suptitle('Measures of Dispersion', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Same mean, different spreads
ax1 = axes[0, 0]
x = np.linspace(50, 150, 200)
for std, color, label in [(5, MLBLUE, 'Std=5 (Low Risk)'),
                           (15, MLORANGE, 'Std=15 (Medium)'),
                           (25, MLRED, 'Std=25 (High Risk)')]:
    y = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-100)/std)**2)
    ax1.plot(x, y, color=color, linewidth=2.5, label=label)
    ax1.fill_between(x, y, alpha=0.2, color=color)

ax1.axvline(100, color='black', linestyle='--', linewidth=1)
ax1.set_title('Same Mean (100), Different Standard Deviations', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Return (%)', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Variance calculation visual
ax2 = axes[0, 1]
data = np.array([85, 90, 95, 100, 105, 110, 115])
mean = np.mean(data)
deviations = data - mean
ax2.bar(range(len(data)), data, color=MLBLUE, alpha=0.7, edgecolor='black')
ax2.axhline(mean, color=MLRED, linewidth=2, linestyle='--', label=f'Mean = {mean:.0f}')

# Show deviations
for i, (d, dev) in enumerate(zip(data, deviations)):
    color = MLGREEN if dev >= 0 else MLRED
    ax2.annotate('', xy=(i, mean), xytext=(i, d),
                 arrowprops=dict(arrowstyle='<->', color=color, lw=2))
    ax2.text(i + 0.1, (d + mean)/2, f'{dev:+.0f}', fontsize=8, color=color, fontweight='bold')

ax2.set_xticks(range(len(data)))
ax2.set_xticklabels([f'x{i+1}' for i in range(len(data))], fontsize=9)
ax2.set_title(f'Deviations from Mean (Var = {np.var(data):.0f})', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Observation', fontsize=10)
ax2.set_ylabel('Value', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Stock volatility comparison
ax3 = axes[1, 0]
days = 100
stock_low = 100 + np.cumsum(np.random.randn(days) * 0.5)
stock_high = 100 + np.cumsum(np.random.randn(days) * 2.5)

ax3.plot(stock_low, color=MLGREEN, linewidth=2, label=f'Low Vol (Std={np.std(np.diff(stock_low)):.2f})')
ax3.plot(stock_high, color=MLRED, linewidth=2, label=f'High Vol (Std={np.std(np.diff(stock_high)):.2f})')
ax3.axhline(100, color='gray', linestyle=':', linewidth=1)
ax3.set_title('Two Stocks: Same Starting Point, Different Volatility', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Trading Day', fontsize=10)
ax3.set_ylabel('Price ($)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Formula visualization
ax4 = axes[1, 1]
ax4.axis('off')

formulas = [
    ('Range', 'max(x) - min(x)', 'Simplest measure, sensitive to outliers', MLBLUE),
    ('Variance (population)', 'Var = (1/N) * sum((xi - mean)^2)', 'Average squared deviation', MLORANGE),
    ('Standard Deviation', 'Std = sqrt(Var)', 'Same units as data', MLGREEN),
    ('Coefficient of Variation', 'CV = Std / Mean', 'Relative dispersion', MLRED),
]

ax4.text(0.5, 0.95, 'Dispersion Measures', ha='center', fontsize=14, fontweight='bold',
         color=MLPURPLE, transform=ax4.transAxes)

y = 0.8
for name, formula, desc, color in formulas:
    ax4.text(0.05, y, name + ':', fontsize=11, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.35, y, formula, fontsize=10, family='monospace', transform=ax4.transAxes)
    ax4.text(0.05, y - 0.05, desc, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.2

# Finance context
ax4.text(0.5, 0.1, 'In Finance: Std Dev = Volatility = Risk', ha='center',
         fontsize=12, fontweight='bold', color=MLPURPLE, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
