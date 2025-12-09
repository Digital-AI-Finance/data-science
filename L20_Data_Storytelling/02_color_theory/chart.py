"""Color Theory - Strategic use of color in visualizations"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
fig.suptitle('Strategic Color Use in Data Visualization', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Sequential palette (gradients for ordered data)
ax1 = axes[0, 0]

# Show gradient from light to dark blue
gradient = np.linspace(0, 1, 10)
cmap = plt.cm.Blues

for i, val in enumerate(gradient):
    ax1.bar(i, 1, color=cmap(0.2 + val * 0.7), width=0.9, edgecolor='white')
    ax1.text(i, 1.1, f'{int(val*100)}%', ha='center', fontsize=8)

ax1.set_xlim(-0.5, 9.5)
ax1.set_ylim(0, 1.5)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Sequential: Low to High Values', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add use case annotation
ax1.text(4.5, -0.2, 'Use for: Magnitude, Percentages, Concentration',
         ha='center', fontsize=9, style='italic', color='gray')

# Plot 2: Diverging palette (values above/below center)
ax2 = axes[0, 1]

values = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
cmap_div = plt.cm.RdYlGn

for i, val in enumerate(values):
    # Map values from -4 to 4 onto 0 to 1
    color_val = (val + 4) / 8
    ax2.bar(i, abs(val) + 0.5, color=cmap_div(color_val), width=0.9, edgecolor='white')
    ax2.text(i, abs(val) + 0.7, f'{val:+d}%', ha='center', fontsize=9)

ax2.axhline(0.5, color='gray', linestyle='-', linewidth=1)
ax2.set_xlim(-0.5, 8.5)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('Diverging: Negative to Positive', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.text(4, -0.5, 'Use for: Returns, Changes, Deviations from baseline',
         ha='center', fontsize=9, style='italic', color='gray')

# Plot 3: Strategic emphasis with color
ax3 = axes[1, 0]

categories = ['Tech', 'Finance', 'Health', 'Energy', 'Consumer', 'Utilities']
values = [15.2, 8.5, 12.1, 6.3, 9.8, 4.2]

# Emphasize one category
colors = [MLLAVENDER] * 6
colors[0] = MLBLUE  # Emphasize Tech

bars = ax3.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)

# Add emphasis annotation
ax3.annotate('Focus: Tech sector\noutperforms all others', xy=(0, 15.2), xytext=(2, 17),
             fontsize=10, fontweight='bold', color=MLBLUE,
             arrowprops=dict(arrowstyle='->', color=MLBLUE, lw=2))

ax3.set_title('Emphasis: Draw Attention with Color', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(alpha=0.3, axis='y')

# Plot 4: Color best practices
ax4 = axes[1, 1]
ax4.axis('off')

# Show do's and don'ts
practices = [
    ('DO', [
        'Use red/green for negative/positive',
        'Limit palette to 5-7 colors',
        'Use gray for context/background',
        'Test for colorblind accessibility'
    ], MLGREEN),
    ('DON\'T', [
        'Use rainbow palettes for sequential data',
        'Mix too many saturated colors',
        'Rely only on color for encoding',
        'Use red for neutral information'
    ], MLRED)
]

for col, (title, items, color) in enumerate(practices):
    x = 0.05 + col * 0.5
    ax4.text(x + 0.2, 0.95, title, fontsize=14, fontweight='bold', color=color, ha='center')
    for i, item in enumerate(items):
        ax4.text(x, 0.8 - i * 0.15, f'{"+" if title=="DO" else "-"} {item}',
                fontsize=9, color='black' if title == 'DO' else 'gray')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Color Best Practices', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
