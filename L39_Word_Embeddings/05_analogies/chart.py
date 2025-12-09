"""Word Analogies - Vector Arithmetic"""
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
fig.suptitle('Word Analogies: Vector Arithmetic', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Analogy concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WORD ANALOGIES

THE DISCOVERY:
--------------
Word embeddings can do arithmetic!

king - man + woman = queen

The relationship "man is to woman" is captured
as a consistent direction in vector space.


HOW IT WORKS:
-------------
vec("king") - vec("man") = relationship "royalty"
Add vec("woman") = female royalty = "queen"


FAMOUS EXAMPLES:
----------------
king - man + woman = queen
paris - france + italy = rome
walked - walk + swim = swam
bigger - big + small = smaller


FORMULA:
--------
A is to B as C is to ?

? = vec(B) - vec(A) + vec(C)

Then find the word closest to this result.


IN GENSIM:
----------
result = model.wv.most_similar(
    positive=['woman', 'king'],
    negative=['man'],
    topn=1
)
# Returns: [('queen', 0.85)]


WHY THIS MATTERS:
-----------------
1. Shows embeddings capture semantics
2. Enables reasoning with vectors
3. Useful for data augmentation
4. Test embedding quality
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Analogy Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual analogy
ax2 = axes[0, 1]
ax2.axis('off')

# Draw analogy as vector diagram
# Simplified 2D representation
points = {
    'man': (1, 1),
    'woman': (1, 3),
    'king': (4, 1),
    'queen': (4, 3)
}

# Draw points
for word, (x, y) in points.items():
    color = MLBLUE if 'man' in word or 'woman' in word else MLGREEN
    ax2.scatter(x, y, c=color, s=200, zorder=5)
    ax2.text(x + 0.15, y + 0.15, word, fontsize=11, fontweight='bold')

# Draw vectors
# man -> woman (gender)
ax2.annotate('', xy=(1, 2.8), xytext=(1, 1.2),
            arrowprops=dict(arrowstyle='->', color=MLRED, lw=2))
ax2.text(0.5, 2, 'gender', fontsize=9, color=MLRED, rotation=90)

# king -> queen (gender)
ax2.annotate('', xy=(4, 2.8), xytext=(4, 1.2),
            arrowprops=dict(arrowstyle='->', color=MLRED, lw=2))

# man -> king (royalty)
ax2.annotate('', xy=(3.8, 1), xytext=(1.2, 1),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
ax2.text(2.5, 0.7, 'royalty', fontsize=9, color=MLORANGE)

# woman -> queen (royalty)
ax2.annotate('', xy=(3.8, 3), xytext=(1.2, 3),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))

# Formula
ax2.text(2.5, 4.2, 'king - man + woman = queen', fontsize=12, ha='center',
         fontweight='bold', bbox=dict(facecolor=MLLAVENDER, alpha=0.8))

ax2.set_xlim(0, 5)
ax2.set_ylim(0, 4.5)
ax2.set_title('Analogy Visualization', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Finance analogies
ax3 = axes[1, 0]
ax3.axis('off')

finance = '''
FINANCIAL ANALOGIES

COMPANY RELATIONSHIPS:
----------------------
apple - technology + finance = ?
  -> goldman, jpmorgan

google - search + social = ?
  -> facebook, twitter


CONCEPT ANALOGIES:
------------------
profit - revenue + cost = ?
  -> loss, expense

bullish - positive + negative = ?
  -> bearish

stock - equity + debt = ?
  -> bond


SECTOR ANALOGIES:
-----------------
microsoft - software + oil = ?
  -> exxon, chevron


TESTING ON YOUR EMBEDDINGS:
---------------------------
# Test if embeddings capture financial concepts
def test_analogy(model, a, b, c, expected):
    result = model.wv.most_similar(
        positive=[b, c],
        negative=[a],
        topn=1
    )[0]
    print(f"{a} - {b} + {c} = {result[0]} ({result[1]:.2f})")
    return result[0] == expected

# Run tests
test_analogy(model, 'stock', 'equity', 'debt', 'bond')
test_analogy(model, 'profit', 'positive', 'negative', 'loss')


ANALOGY ACCURACY:
-----------------
Good embeddings: 40-60% accuracy on standard tests
Domain-specific: Higher on in-domain analogies
'''

ax3.text(0.02, 0.98, finance, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Financial Analogies', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Limitations
ax4 = axes[1, 1]
ax4.axis('off')

limitations = '''
LIMITATIONS OF ANALOGIES

NOT ALWAYS ACCURATE:
--------------------
Word embeddings don't always give correct analogies.

king - man + woman = queen  (works!)
doctor - man + woman = nurse  (biased!)


BIAS IN EMBEDDINGS:
-------------------
Embeddings learn from data, including biases.

man - programmer + homemaker = woman (oops!)

Be aware of:
- Gender bias
- Racial bias
- Stereotypes


DOMAIN SPECIFICITY:
-------------------
General embeddings may not work for finance.

"Apple" in general text = fruit
"Apple" in financial text = company

Solution: Train domain-specific embeddings!


ANALOGY FAILURES:
-----------------
- Rare words have poor vectors
- Multiple meanings confuse the math
- Some relationships aren't linear


BEST PRACTICES:
---------------
1. Test analogies on your domain
2. Use multiple top results, not just #1
3. Be aware of bias
4. Combine with other methods


FOR SERIOUS NLP:
----------------
Modern transformers (BERT, GPT) handle
analogies better through contextual embeddings.
'''

ax4.text(0.02, 0.98, limitations, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Limitations', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
