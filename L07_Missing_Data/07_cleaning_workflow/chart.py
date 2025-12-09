"""Data Cleaning Workflow - Flowchart showing the complete cleaning process"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
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

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Data Cleaning Workflow for Financial Data',
        ha='center', fontsize=16, fontweight='bold', color=MLPURPLE)

# Define workflow steps
steps = [
    {'x': 2, 'y': 7.5, 'text': '1. Load Data', 'detail': 'pd.read_csv()', 'color': MLBLUE},
    {'x': 5, 'y': 7.5, 'text': '2. Inspect', 'detail': 'df.info()\ndf.head()', 'color': MLBLUE},
    {'x': 8, 'y': 7.5, 'text': '3. Check Missing', 'detail': 'df.isnull().sum()', 'color': MLORANGE},
    {'x': 11, 'y': 7.5, 'text': '4. Handle Missing', 'detail': 'fillna() / dropna()', 'color': MLORANGE},
    {'x': 2, 'y': 4.5, 'text': '5. Check Duplicates', 'detail': 'df.duplicated()', 'color': MLRED},
    {'x': 5, 'y': 4.5, 'text': '6. Remove Duplicates', 'detail': 'drop_duplicates()', 'color': MLRED},
    {'x': 8, 'y': 4.5, 'text': '7. Fix Data Types', 'detail': 'astype() / to_datetime()', 'color': MLGREEN},
    {'x': 11, 'y': 4.5, 'text': '8. Validate', 'detail': 'df.describe()', 'color': MLGREEN},
]

# Draw boxes and text
for step in steps:
    box = FancyBboxPatch((step['x']-1.3, step['y']-0.6), 2.6, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor=step['color'], facecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(step['x'], step['y']+0.2, step['text'], ha='center', va='center',
            fontsize=10, fontweight='bold', color=step['color'])
    ax.text(step['x'], step['y']-0.25, step['detail'], ha='center', va='center',
            fontsize=8, family='monospace', color='gray')

# Draw arrows
arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=8"
arrows = [
    ((3.3, 7.5), (3.7, 7.5)),  # 1 -> 2
    ((6.3, 7.5), (6.7, 7.5)),  # 2 -> 3
    ((9.3, 7.5), (9.7, 7.5)),  # 3 -> 4
    ((11, 6.9), (11, 5.7)),    # 4 -> 8 (down)
    ((11, 5.1), (9.3, 5.1)),   # 8 -> 7 (left, top row)
    ((6.7, 4.5), (6.3, 4.5)),  # 7 -> 6 (left)
    ((3.7, 4.5), (3.3, 4.5)),  # 6 -> 5 (left)
]

for start, end in arrows[:4]:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15,
                            color=MLPURPLE, linewidth=2)
    ax.add_patch(arrow)

# Curved arrow from step 4 down to step 8
ax.annotate('', xy=(11, 5.1), xytext=(11, 6.9),
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2,
                           connectionstyle='arc3,rad=0'))

# Horizontal arrows on bottom row (right to left)
for i, (start, end) in enumerate(arrows[4:]):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))

# Final output box
output_box = FancyBboxPatch((0.5, 1.5), 4, 1.5, boxstyle="round,pad=0.1",
                            edgecolor=MLGREEN, facecolor='#E8FFE8', linewidth=2)
ax.add_patch(output_box)
ax.text(2.5, 2.5, 'Clean Data', ha='center', va='center',
        fontsize=12, fontweight='bold', color=MLGREEN)
ax.text(2.5, 1.9, 'Ready for Analysis', ha='center', va='center',
        fontsize=9, style='italic', color='gray')

# Arrow from step 5 to output
ax.annotate('', xy=(2.5, 3.0), xytext=(2, 3.9),
            arrowprops=dict(arrowstyle='->', color=MLGREEN, lw=2))

# Quality metrics box
metrics_box = FancyBboxPatch((6, 1.5), 7, 1.5, boxstyle="round,pad=0.1",
                             edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3)
ax.add_patch(metrics_box)
ax.text(9.5, 2.6, 'Quality Metrics to Track:', ha='center', va='center',
        fontsize=10, fontweight='bold', color=MLPURPLE)
ax.text(9.5, 1.9, 'Missing % | Duplicate % | Data Type Errors | Value Range Violations',
        ha='center', va='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
