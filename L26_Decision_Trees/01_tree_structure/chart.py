"""Tree Structure - How decision trees work"""
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
fig.suptitle('Decision Tree Structure', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Tree diagram
ax1 = axes[0, 0]
ax1.axis('off')

# Draw a simple decision tree
def draw_node(ax, x, y, text, is_leaf=False, color=MLBLUE):
    bbox = dict(boxstyle='round,pad=0.3', facecolor=color if not is_leaf else MLGREEN, alpha=0.8)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, bbox=bbox)

def draw_edge(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2+0.05), xytext=(x1, y1-0.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax.text(mid_x, mid_y, label, fontsize=8, ha='center', color=MLPURPLE)

# Root node
draw_node(ax1, 0.5, 0.9, 'Volume > 1.5M?', color=MLBLUE)

# Level 1
draw_node(ax1, 0.25, 0.6, 'Returns > 0?', color=MLBLUE)
draw_node(ax1, 0.75, 0.6, 'Volatility > 2%?', color=MLBLUE)

# Level 2 - Leaves
draw_node(ax1, 0.1, 0.25, 'SELL\n(65%)', is_leaf=True)
draw_node(ax1, 0.4, 0.25, 'BUY\n(72%)', is_leaf=True)
draw_node(ax1, 0.6, 0.25, 'HOLD\n(58%)', is_leaf=True)
draw_node(ax1, 0.9, 0.25, 'SELL\n(81%)', is_leaf=True)

# Edges
draw_edge(ax1, 0.5, 0.85, 0.25, 0.65, 'No')
draw_edge(ax1, 0.5, 0.85, 0.75, 0.65, 'Yes')
draw_edge(ax1, 0.25, 0.55, 0.1, 0.30, 'No')
draw_edge(ax1, 0.25, 0.55, 0.4, 0.30, 'Yes')
draw_edge(ax1, 0.75, 0.55, 0.6, 0.30, 'No')
draw_edge(ax1, 0.75, 0.55, 0.9, 0.30, 'Yes')

ax1.set_xlim(0, 1)
ax1.set_ylim(0.1, 1)
ax1.set_title('Decision Tree Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Terminology
ax2 = axes[0, 1]
ax2.axis('off')

terminology = '''
DECISION TREE TERMINOLOGY

ROOT NODE
---------
- Top of the tree
- First splitting decision
- Contains ALL training samples

INTERNAL NODES
--------------
- Decision points
- Split on a feature threshold
- "Is feature X > value?"

LEAF NODES (Terminal)
---------------------
- End of a branch
- No more splits
- Final prediction
- Majority class (classification)
- Mean value (regression)

BRANCH / EDGE
-------------
- Connects nodes
- Represents decision outcome
- Yes/No, True/False, Left/Right

DEPTH
-----
- Number of edges from root to leaf
- Deeper = more complex
- Too deep = overfitting

SPLITTING CRITERIA
------------------
Classification: Gini, Entropy
Regression: MSE, MAE
'''

ax2.text(0.02, 0.98, terminology, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Tree Terminology', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Decision boundary visualization
ax3 = axes[1, 0]

# Generate 2D data
n = 200
X0 = np.random.multivariate_normal([2, 5], [[1, 0], [0, 1]], n//2)
X1 = np.random.multivariate_normal([5, 2], [[1, 0], [0, 1]], n//2)

ax3.scatter(X0[:, 0], X0[:, 1], c=MLBLUE, s=30, alpha=0.6, label='Class 0', edgecolors='black')
ax3.scatter(X1[:, 0], X1[:, 1], c=MLORANGE, s=30, alpha=0.6, label='Class 1', edgecolors='black')

# Draw decision boundaries (axis-aligned)
ax3.axhline(3.5, color=MLRED, linewidth=2.5, linestyle='--')
ax3.axvline(3.5, xmin=0, xmax=0.5, color=MLRED, linewidth=2.5, linestyle='--')

# Fill regions
ax3.fill([0, 3.5, 3.5, 0], [0, 0, 3.5, 3.5], alpha=0.15, color=MLORANGE)
ax3.fill([3.5, 7, 7, 3.5], [0, 0, 7, 7], alpha=0.15, color=MLORANGE)
ax3.fill([0, 3.5, 3.5, 0], [3.5, 3.5, 7, 7], alpha=0.15, color=MLBLUE)

ax3.set_title('Decision Tree Creates Axis-Aligned Boundaries', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Feature 1 (Returns)', fontsize=10)
ax3.set_ylabel('Feature 2 (Volume)', fontsize=10)
ax3.set_xlim(0, 7)
ax3.set_ylim(0, 7)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Prediction process
ax4 = axes[1, 1]
ax4.axis('off')

process = '''
HOW PREDICTION WORKS

1. START AT ROOT
   New sample: Volume=2M, Returns=+1.5%, Vol=1.8%

2. EVALUATE CONDITION
   "Volume > 1.5M?" -> YES (2M > 1.5M)
   Go RIGHT

3. EVALUATE NEXT CONDITION
   "Volatility > 2%?" -> NO (1.8% < 2%)
   Go LEFT

4. REACH LEAF NODE
   Prediction: HOLD (58% confidence)


CLASSIFICATION vs REGRESSION
-----------------------------

Classification Tree:
- Leaf = majority class
- Output: class label
- Confidence = class proportion

Regression Tree:
- Leaf = mean of y values
- Output: continuous value
- Can also output variance


KEY INSIGHT:
------------
Trees make predictions by following
a series of if-then-else rules.

Very interpretable!
"If volume is high AND volatility is low,
 then HOLD"
'''

ax4.text(0.02, 0.98, process, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Prediction Process', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
