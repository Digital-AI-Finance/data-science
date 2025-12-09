"""Spurious Correlation - Misleading relationships"""
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
fig.suptitle('Spurious Correlations: When r Lies', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Confounding variable
ax1 = axes[0, 0]
# Ice cream sales and drowning deaths - both caused by hot weather
time = np.arange(12)  # Months
ice_cream = 50 + 30 * np.sin(2 * np.pi * time / 12) + np.random.normal(0, 5, 12)
drownings = 20 + 15 * np.sin(2 * np.pi * time / 12) + np.random.normal(0, 3, 12)

ax1.scatter(ice_cream, drownings, color=MLBLUE, s=80, edgecolors='black')
r = np.corrcoef(ice_cream, drownings)[0, 1]

z = np.polyfit(ice_cream, drownings, 1)
ax1.plot(np.sort(ice_cream), np.poly1d(z)(np.sort(ice_cream)), color=MLRED, linewidth=2)

ax1.set_title(f'Ice Cream vs Drownings (r = {r:.2f})', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Ice Cream Sales', fontsize=10)
ax1.set_ylabel('Drowning Deaths', fontsize=10)
ax1.grid(alpha=0.3)
ax1.text(0.5, 0.05, 'Confounding variable: Summer temperature!',
         transform=ax1.transAxes, fontsize=9, ha='center', style='italic', color=MLRED)

# Plot 2: Simpson's Paradox
ax2 = axes[0, 1]
# Overall negative, but positive within groups
group1_x = np.random.normal(2, 0.5, 30)
group1_y = 0.5 * group1_x + np.random.normal(0, 0.3, 30) + 3

group2_x = np.random.normal(4, 0.5, 30)
group2_y = 0.5 * group2_x + np.random.normal(0, 0.3, 30) + 1

group3_x = np.random.normal(6, 0.5, 30)
group3_y = 0.5 * group3_x + np.random.normal(0, 0.3, 30) - 1

all_x = np.concatenate([group1_x, group2_x, group3_x])
all_y = np.concatenate([group1_y, group2_y, group3_y])

ax2.scatter(group1_x, group1_y, color=MLBLUE, s=50, label='Group 1', alpha=0.7)
ax2.scatter(group2_x, group2_y, color=MLGREEN, s=50, label='Group 2', alpha=0.7)
ax2.scatter(group3_x, group3_y, color=MLORANGE, s=50, label='Group 3', alpha=0.7)

# Group trendlines (positive)
for gx, gy, color in [(group1_x, group1_y, MLBLUE), (group2_x, group2_y, MLGREEN),
                       (group3_x, group3_y, MLORANGE)]:
    z = np.polyfit(gx, gy, 1)
    ax2.plot(np.sort(gx), np.poly1d(z)(np.sort(gx)), color=color, linewidth=2)

# Overall trendline (negative!)
z_all = np.polyfit(all_x, all_y, 1)
ax2.plot(np.sort(all_x), np.poly1d(z_all)(np.sort(all_x)), color=MLRED, linewidth=3, linestyle='--')

r_overall = np.corrcoef(all_x, all_y)[0, 1]
ax2.set_title(f"Simpson's Paradox (Overall r = {r_overall:.2f})", fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Non-linear relationship (zero correlation)
ax3 = axes[1, 0]
x = np.linspace(-3, 3, 100)
y = x**2 + np.random.normal(0, 0.5, 100)
r = np.corrcoef(x, y)[0, 1]

ax3.scatter(x, y, color=MLGREEN, s=50, alpha=0.6, edgecolors='black')
ax3.set_title(f'Non-linear: r = {r:.2f} (but clearly related!)', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Y = X^2 + noise', fontsize=10)
ax3.grid(alpha=0.3)
ax3.text(0.5, 0.95, 'Pearson only measures LINEAR relationship!',
         transform=ax3.transAxes, fontsize=9, ha='center', va='top', style='italic', color=MLRED)

# Plot 4: Warnings
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Correlation Pitfalls', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

warnings = [
    ('Confounding Variables', 'Both X and Y caused by hidden Z', MLBLUE),
    ("Simpson's Paradox", 'Trend reverses when data is grouped', MLGREEN),
    ('Non-linearity', 'r = 0 does not mean no relationship', MLORANGE),
    ('Outliers', 'Single point can dominate correlation', MLRED),
    ('Coincidence', 'Random correlations in large datasets', MLPURPLE),
]

y = 0.78
for title, desc, color in warnings:
    ax4.text(0.1, y, title + ':', fontsize=11, fontweight='bold', color=color, transform=ax4.transAxes)
    ax4.text(0.45, y, desc, fontsize=9, transform=ax4.transAxes)
    y -= 0.13

ax4.text(0.5, 0.1, 'Always visualize your data before trusting correlation!',
         ha='center', fontsize=11, style='italic', color=MLRED, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLRED))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
