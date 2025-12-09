"""Tree Visualization - Visualizing decision trees"""
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
fig.suptitle('Visualizing Decision Trees', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Simple tree structure diagram
ax1 = axes[0, 0]
ax1.axis('off')

# Draw tree manually
def draw_box(ax, x, y, text, color=MLLAVENDER, width=0.15, height=0.1):
    from matplotlib.patches import FancyBboxPatch
    bbox = FancyBboxPatch((x-width/2, y-height/2), width, height,
                          boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(bbox)
    ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold')

# Root
draw_box(ax1, 0.5, 0.85, 'RSI <= 30?\nGini=0.45\nn=200', color=MLBLUE)

# Level 1
draw_box(ax1, 0.25, 0.55, 'Vol <= 2%?\nGini=0.35\nn=80', color=MLBLUE)
draw_box(ax1, 0.75, 0.55, 'Mom > 0?\nGini=0.40\nn=120', color=MLBLUE)

# Level 2 (leaves)
draw_box(ax1, 0.1, 0.25, 'BUY\nGini=0.10\nn=50', color=MLGREEN)
draw_box(ax1, 0.35, 0.25, 'HOLD\nGini=0.20\nn=30', color=MLORANGE)
draw_box(ax1, 0.6, 0.25, 'HOLD\nGini=0.15\nn=65', color=MLORANGE)
draw_box(ax1, 0.9, 0.25, 'SELL\nGini=0.08\nn=55', color=MLRED)

# Draw edges
edges = [
    (0.5, 0.80, 0.25, 0.60, 'True'),
    (0.5, 0.80, 0.75, 0.60, 'False'),
    (0.25, 0.50, 0.1, 0.30, 'True'),
    (0.25, 0.50, 0.35, 0.30, 'False'),
    (0.75, 0.50, 0.6, 0.30, 'False'),
    (0.75, 0.50, 0.9, 0.30, 'True')
]

for x1, y1, x2, y2, label in edges:
    ax1.annotate('', xy=(x2, y2+0.05), xytext=(x1, y1-0.05),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax1.text(mid_x - 0.05, mid_y, label, fontsize=7, color=MLPURPLE)

ax1.set_xlim(0, 1)
ax1.set_ylim(0.1, 0.95)
ax1.set_title('Tree Structure (Depth=2)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: sklearn plot_tree code
ax2 = axes[0, 1]
ax2.axis('off')

code = '''
VISUALIZING TREES WITH SKLEARN

from sklearn.tree import plot_tree, export_text
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Train tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Method 1: matplotlib plot
fig, ax = plt.subplots(figsize=(15, 10))
plot_tree(
    clf,
    feature_names=['RSI', 'Vol', 'Mom', 'Volume'],
    class_names=['Sell', 'Hold', 'Buy'],
    filled=True,           # Color by class
    rounded=True,          # Rounded boxes
    fontsize=10,
    ax=ax
)
plt.tight_layout()
plt.savefig('tree.png', dpi=150)

# Method 2: Text representation
rules = export_text(
    clf,
    feature_names=['RSI', 'Vol', 'Mom', 'Volume']
)
print(rules)

Output:
|--- RSI <= 30.00
|   |--- Vol <= 2.00
|   |   |--- class: Buy
|   |--- Vol >  2.00
|   |   |--- class: Hold
|--- RSI >  30.00
|   |--- Mom <= 0.00
|   |   |--- class: Hold
|   |--- Mom >  0.00
|   |   |--- class: Sell
'''

ax2.text(0.02, 0.98, code, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('sklearn Visualization Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Decision surface
ax3 = axes[1, 0]

# Create 2D decision surface
xx, yy = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 5, 100))

# Simulated decision regions for a tree
Z = np.zeros_like(xx)
Z[(xx <= 30) & (yy <= 2)] = 2  # Buy
Z[(xx <= 30) & (yy > 2)] = 1   # Hold
Z[(xx > 30) & (yy <= 3)] = 1   # Hold
Z[(xx > 30) & (yy > 3)] = 0    # Sell

ax3.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
             colors=[MLRED, MLORANGE, MLGREEN], alpha=0.6)
ax3.contour(xx, yy, Z, levels=[0.5, 1.5], colors=['black'], linewidths=2)

# Add sample points
np.random.seed(42)
n = 50
for label, color, region in [(0, MLRED, [(60, 80), (3.5, 5)]),
                              (1, MLORANGE, [(40, 60), (2, 4)]),
                              (2, MLGREEN, [(10, 30), (0.5, 2)])]:
    x = np.random.uniform(region[0][0], region[0][1], n//3)
    y = np.random.uniform(region[1][0], region[1][1], n//3)
    ax3.scatter(x, y, c=color, s=30, edgecolors='black', alpha=0.8)

ax3.set_title('Decision Regions (2D)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('RSI', fontsize=10)
ax3.set_ylabel('Volatility (%)', fontsize=10)

# Add decision boundary labels
ax3.text(15, 1, 'BUY', fontsize=12, fontweight='bold', color='white')
ax3.text(50, 1.5, 'HOLD', fontsize=12, fontweight='bold', color='black')
ax3.text(70, 4, 'SELL', fontsize=12, fontweight='bold', color='white')

# Plot 4: Interpretability summary
ax4 = axes[1, 1]
ax4.axis('off')

summary = '''
WHY VISUALIZE DECISION TREES?

1. INTERPRETABILITY
   - See exactly why predictions are made
   - Explain to stakeholders
   - Regulatory compliance (finance!)

2. MODEL DEBUGGING
   - Identify overfitting (too many splits)
   - Find counterintuitive rules
   - Verify feature usage

3. FEATURE INSIGHTS
   - Which features are used first?
   - What thresholds matter?
   - Interaction between features

4. DOCUMENTATION
   - Export rules for manual implementation
   - Create decision flowcharts
   - Training materials


BEST PRACTICES:
---------------
- Limit depth for visualization (max_depth=3-5)
- Use meaningful feature/class names
- Color-code by class or impurity
- Export both visual and text versions
- Save high-resolution images (dpi=150+)
'''

ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Why Visualization Matters', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
