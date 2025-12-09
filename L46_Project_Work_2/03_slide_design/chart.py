"""Slide Design - Creating Effective Visual Slides"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Slide Design: Creating Effective Visual Slides', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Good vs bad design
ax1 = axes[0, 0]
ax1.axis('off')

# Draw bad slide example
ax1.add_patch(plt.Rectangle((0.02, 0.52), 0.45, 0.45, facecolor=MLRED, alpha=0.1, edgecolor=MLRED))
ax1.text(0.245, 0.92, 'BAD SLIDE', fontsize=10, ha='center', fontweight='bold', color=MLRED)
ax1.text(0.04, 0.85, 'Results of My Analysis of Stock Data', fontsize=7, fontweight='bold')
ax1.text(0.04, 0.8, 'The model achieved 0.85 accuracy on the test set which was\n'
                    'calculated using a train-test split of 80/20. The features\n'
                    'used were: price, volume, returns, moving average, etc...\n'
                    'I also tried Random Forest which got 0.82 and also Linear\n'
                    'Regression which got 0.75. The best model was clearly the\n'
                    'neural network with 3 hidden layers and dropout of 0.2...',
         fontsize=5.5)

# Draw good slide example
ax1.add_patch(plt.Rectangle((0.52, 0.52), 0.45, 0.45, facecolor=MLGREEN, alpha=0.1, edgecolor=MLGREEN))
ax1.text(0.745, 0.92, 'GOOD SLIDE', fontsize=10, ha='center', fontweight='bold', color=MLGREEN)
ax1.text(0.54, 0.85, 'Neural Network: Best Model', fontsize=8, fontweight='bold')
ax1.text(0.54, 0.78, '- Accuracy: 85%', fontsize=7)
ax1.text(0.54, 0.73, '- Beats baseline by 10%', fontsize=7)
ax1.text(0.54, 0.68, '- Key feature: Price momentum', fontsize=7)

# Mini chart in good slide
x_mini = np.array([0.58, 0.68, 0.78, 0.88])
y_mini = np.array([0.6, 0.62, 0.64, 0.63])
ax1.bar(x_mini, [0.75, 0.82, 0.85, 0], width=0.08, color=[MLBLUE, MLORANGE, MLGREEN, 'white'], alpha=0.7)
ax1.text(0.73, 0.56, '[Chart showing comparison]', fontsize=6, ha='center', style='italic')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('Good vs Bad Slides', fontsize=11, fontweight='bold', color=MLPURPLE)

# Design principles text at bottom
ax1.text(0.5, 0.4, 'DESIGN PRINCIPLES', fontsize=10, ha='center', fontweight='bold')
principles = [
    '- Maximum 3-4 bullet points per slide',
    '- Use large fonts (24pt minimum)',
    '- One chart/visual per slide',
    '- White space is your friend',
    '- Consistent colors throughout'
]
for i, p in enumerate(principles):
    ax1.text(0.5, 0.32 - i*0.07, p, fontsize=8, ha='center')

# Plot 2: Chart design tips
ax2 = axes[0, 1]
ax2.axis('off')

chart_tips = '''
CHART DESIGN FOR PRESENTATIONS

CHOOSE THE RIGHT CHART:
-----------------------
- Bar: Compare categories
- Line: Show trends over time
- Pie: Show proportions (max 5 slices)
- Scatter: Show relationships
- Table: Exact values needed


FORMATTING TIPS:
----------------
- Large axis labels (readable from back)
- Clear title explaining the point
- Legend if multiple series
- Remove unnecessary gridlines
- Highlight key data points


COLOR GUIDELINES:
-----------------
- Use 2-3 colors maximum
- Consistent color meanings
- Colorblind-friendly palettes
- Highlight important data
- Gray for context, color for focus


CHART TITLES:
-------------
BAD:  "Figure 1: Model Results"
GOOD: "Neural Network Achieves 85% Accuracy"

BAD:  "Chart showing data"
GOOD: "Stock Returns Increased 15% in Q4"


COMMON MISTAKES:
----------------
- 3D charts (don't use!)
- Too many data series
- Unreadable axis labels
- Missing units
- Misleading scales


EXPORT FOR SLIDES:
------------------
- Save as PNG (300 DPI)
- Test on projector if possible
- Check contrast and readability
'''

ax2.text(0.02, 0.98, chart_tips, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Chart Design Tips', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Font and color guidelines
ax3 = axes[1, 0]
ax3.axis('off')

# Font sizes visual
ax3.text(0.5, 0.95, 'TYPOGRAPHY HIERARCHY', fontsize=12, ha='center', fontweight='bold')

font_examples = [
    ('Slide Title', 28, MLPURPLE),
    ('Section Header', 24, MLBLUE),
    ('Body Text', 18, 'black'),
    ('Minimum Size', 14, 'gray')
]

for i, (text, size, color) in enumerate(font_examples):
    y = 0.8 - i * 0.15
    ax3.text(0.05, y, f'{size}pt:', fontsize=10, color='gray')
    ax3.text(0.2, y, text, fontsize=size/2.5, color=color, fontweight='bold' if i < 2 else 'normal')

# Color palette
ax3.text(0.5, 0.28, 'RECOMMENDED COLOR PALETTE', fontsize=10, ha='center', fontweight='bold')
colors_demo = [MLPURPLE, MLBLUE, MLGREEN, MLORANGE, MLRED]
labels = ['Primary', 'Secondary', 'Success', 'Warning', 'Alert']

for i, (c, l) in enumerate(zip(colors_demo, labels)):
    x = 0.1 + i * 0.18
    ax3.add_patch(plt.Rectangle((x, 0.1), 0.12, 0.12, facecolor=c))
    ax3.text(x + 0.06, 0.06, l, fontsize=7, ha='center')

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title('Typography & Colors', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Slide templates
ax4 = axes[1, 1]
ax4.axis('off')

templates = '''
SLIDE TEMPLATES

TITLE SLIDE:
------------
+-------------------------+
|                         |
|    PROJECT TITLE        |
|                         |
|    Your Name            |
|    Date                 |
|                         |
+-------------------------+


CONTENT + CHART:
----------------
+-------------------------+
| Section Title           |
+-------------------------+
|  - Point 1    |         |
|  - Point 2    | [CHART] |
|  - Point 3    |         |
+-------------------------+


FULL CHART:
-----------
+-------------------------+
| Chart Title (Insight)   |
+-------------------------+
|                         |
|       [BIG CHART]       |
|                         |
+-------------------------+


COMPARISON:
-----------
+-------------------------+
| Before vs After         |
+-------------------------+
|  [Chart A] | [Chart B]  |
|            |            |
+-------------------------+


DEMO SLIDE:
-----------
+-------------------------+
|   LIVE DEMO             |
+-------------------------+
|                         |
|   [App Screenshot]      |
|                         |
|   URL: your-app.com     |
+-------------------------+
'''

ax4.text(0.02, 0.98, templates, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Slide Templates', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
