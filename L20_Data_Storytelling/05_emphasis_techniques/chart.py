"""Emphasis Techniques - Drawing attention to key insights"""
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
fig.suptitle('Emphasis Techniques: Direct Attention', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Color emphasis
ax1 = axes[0, 0]
categories = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
values = [15.2, 12.8, 10.5, 8.4, 28.5, 14.1]

# De-emphasize all except the key insight
colors = [MLLAVENDER] * 6
colors[4] = MLGREEN  # Emphasize TSLA

bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)

# Add emphasis annotation
ax1.annotate('TSLA: +28.5%\nHighest return', xy=(4, 28.5), xytext=(4, 35),
             fontsize=11, fontweight='bold', color=MLGREEN, ha='center',
             arrowprops=dict(arrowstyle='->', color=MLGREEN, lw=2))

ax1.set_title('Technique 1: Color Contrast', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Return (%)', fontsize=10)
ax1.set_ylim(0, 40)
ax1.grid(alpha=0.3, axis='y')

# Plot 2: Size emphasis
ax2 = axes[0, 1]
categories = ['Q1', 'Q2', 'Q3', 'Q4']
values = [45, 52, 78, 61]  # Q3 is the highlight

# Variable bar width for emphasis
widths = [0.6, 0.6, 0.9, 0.6]
colors = [MLBLUE, MLBLUE, MLORANGE, MLBLUE]

for i, (cat, val, w, c) in enumerate(zip(categories, values, widths, colors)):
    ax2.bar(i, val, width=w, color=c, edgecolor='black', linewidth=0.5 if w == 0.6 else 1.5)
    if w == 0.9:  # Emphasized bar
        ax2.text(i, val + 3, f'${val}M', ha='center', fontsize=12, fontweight='bold', color=MLORANGE)
    else:
        ax2.text(i, val + 3, f'${val}M', ha='center', fontsize=9, color='gray')

ax2.set_xticks(range(4))
ax2.set_xticklabels(categories)
ax2.set_title('Technique 2: Size & Weight', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Revenue ($M)', fontsize=10)
ax2.set_ylim(0, 95)
ax2.grid(alpha=0.3, axis='y')

# Plot 3: Position emphasis (eye-tracking pattern)
ax3 = axes[1, 0]
np.random.seed(42)

# Create scatter with one emphasized point
x = np.random.uniform(5, 25, 30)
y = np.random.uniform(2, 15, 30)

# Add the key insight point
key_x, key_y = 22, 14

ax3.scatter(x, y, c=MLLAVENDER, s=60, alpha=0.6, edgecolors='gray')
ax3.scatter([key_x], [key_y], c=MLGREEN, s=200, edgecolors='black', linewidth=2, zorder=5)

# Add emphasis elements
circle = plt.Circle((key_x, key_y), 2.5, fill=False, color=MLGREEN, linewidth=2, linestyle='--')
ax3.add_patch(circle)

ax3.annotate('Outlier:\nHigh return, Low risk', xy=(key_x, key_y), xytext=(key_x - 8, key_y + 2),
             fontsize=10, fontweight='bold', color=MLGREEN,
             arrowprops=dict(arrowstyle='->', color=MLGREEN, lw=2))

ax3.set_title('Technique 3: Position & Isolation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Risk (%)', fontsize=10)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: Before/After comparison showing emphasis impact
ax4 = axes[1, 1]
ax4.axis('off')

# Show the concept
emphasis_guide = '''EMPHASIS HIERARCHY

1. COLOR CONTRAST
   - Use bright/saturated for emphasis
   - Gray out non-essential elements
   - Limit emphasis to 1-2 items

2. SIZE VARIATION
   - Larger = more important
   - Bold text for key numbers
   - Thicker lines for main series

3. POSITION
   - Top-left draws first attention
   - Isolate key points
   - Use whitespace strategically

4. ANNOTATION
   - Call out the insight
   - Use arrows to direct eyes
   - Keep it concise

RULE: If everything is emphasized,
      nothing is emphasized.
'''

ax4.text(0.05, 0.95, emphasis_guide, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Emphasis Principles', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
