"""Storytelling Flow - Narrative structure in data visualization"""
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
fig.suptitle('Data Storytelling: Narrative Structure', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Story arc visualization
ax1 = axes[0, 0]
ax1.axis('off')

# Draw story arc
arc_x = np.linspace(0, 10, 100)
arc_y = 2 + 3 * np.sin(arc_x * np.pi / 10)

ax1.plot(arc_x, arc_y, color=MLBLUE, linewidth=3)

# Mark story points
story_points = [
    (0.5, 2.4, 'Setup', 'Context &\nBackground'),
    (2.5, 4.2, 'Rising Action', 'Build\nTension'),
    (5, 5, 'Climax', 'Key\nInsight'),
    (7.5, 4.2, 'Falling Action', 'Support\nEvidence'),
    (9.5, 2.4, 'Resolution', 'Call to\nAction')
]

for x, y, label, desc in story_points:
    ax1.scatter([x], [y], s=150, c=MLORANGE, zorder=5, edgecolors='black')
    ax1.text(x, y + 0.6, label, ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)
    ax1.text(x, y - 0.5, desc, ha='center', fontsize=8, color='gray')

ax1.set_xlim(-0.5, 10.5)
ax1.set_ylim(0, 6.5)
ax1.set_title('The Data Story Arc', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Before (boring) vs After (story)
ax2 = axes[0, 1]

# Create "boring" chart that tells no story
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
revenue = [45, 52, 48, 61]

# Show with story elements
bars = ax2.bar(quarters, revenue, color=[MLLAVENDER, MLLAVENDER, MLRED, MLGREEN],
               edgecolor='black', linewidth=0.5)
ax2.axhline(50, color='gray', linestyle='--', linewidth=1.5, label='Target: $50M')

# Add storytelling annotations
ax2.annotate('New CEO\nappointed', xy=(2, 48), xytext=(2.5, 40),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color=MLRED))
ax2.annotate('Strategy pays off!\n+27% QoQ', xy=(3, 61), xytext=(2.8, 68),
             fontsize=9, ha='center', fontweight='bold', color=MLGREEN,
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

ax2.set_title('With Story: Context + Insight', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Revenue ($M)', fontsize=10)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 75)

# Plot 3: Key principles
ax3 = axes[1, 0]
ax3.axis('off')

principles = [
    ('1. Lead with the Insight', 'Start with your key finding, not the methodology', MLBLUE),
    ('2. Provide Context', 'Show why the data matters to the audience', MLGREEN),
    ('3. Build Progressively', 'Layer information, don\'t overwhelm', MLORANGE),
    ('4. Use Contrast', 'Highlight what changed or differs', MLRED),
    ('5. End with Action', 'Tell the audience what to do next', MLPURPLE)
]

for i, (title, desc, color) in enumerate(principles):
    y = 0.85 - i * 0.18
    ax3.add_patch(mpatches.Rectangle((0.02, y - 0.06), 0.04, 0.12, color=color, alpha=0.8))
    ax3.text(0.08, y, title, fontsize=11, fontweight='bold', va='center', color=MLPURPLE)
    ax3.text(0.08, y - 0.06, desc, fontsize=9, va='top', color='gray')

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title('5 Principles of Data Storytelling', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Flow example - Sequential revelation
ax4 = axes[1, 1]

days = np.arange(1, 31)
stock_price = 100 + np.cumsum(np.random.randn(30) * 2)

# Show progressive revelation
ax4.plot(days[:10], stock_price[:10], color=MLLAVENDER, linewidth=2, label='Context (past)')
ax4.plot(days[9:20], stock_price[9:20], color=MLBLUE, linewidth=3, label='Build-up')
ax4.plot(days[19:], stock_price[19:], color=MLGREEN, linewidth=3, label='Resolution')

# Add story markers
ax4.scatter([10], [stock_price[9]], s=100, c=MLBLUE, zorder=5, edgecolors='black')
ax4.scatter([20], [stock_price[19]], s=150, c=MLORANGE, zorder=5, edgecolors='black')
ax4.scatter([30], [stock_price[29]], s=150, c=MLGREEN, zorder=5, edgecolors='black')

ax4.annotate('Earnings\nRelease', xy=(10, stock_price[9]), xytext=(12, stock_price[9]-5),
             fontsize=9, arrowprops=dict(arrowstyle='->', color=MLBLUE))
ax4.annotate('Key\nInsight', xy=(20, stock_price[19]), xytext=(17, stock_price[19]+5),
             fontsize=9, fontweight='bold', color=MLORANGE,
             arrowprops=dict(arrowstyle='->', color=MLORANGE))

ax4.set_title('Sequential Data Revelation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Day', fontsize=10)
ax4.set_ylabel('Price ($)', fontsize=10)
ax4.legend(fontsize=8, loc='lower right')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
