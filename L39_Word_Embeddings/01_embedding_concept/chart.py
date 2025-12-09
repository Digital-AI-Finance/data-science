"""Word Embedding Concept - Dense Vector Representations"""
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
fig.suptitle('Word Embeddings: Dense Vector Representations', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT ARE WORD EMBEDDINGS?

DEFINITION:
-----------
Dense, low-dimensional vector representations
of words that capture semantic meaning.


ONE-HOT vs EMBEDDINGS:
----------------------
One-hot (sparse):
  "stock" = [0, 0, 1, 0, 0, ..., 0]  (50,000 dimensions!)
  - No semantic information
  - Huge dimensionality
  - Every word equally different

Embedding (dense):
  "stock" = [0.23, -0.41, 0.87, ...]  (100-300 dimensions)
  - Captures meaning
  - Compact representation
  - Similar words have similar vectors


KEY PROPERTY:
-------------
Words with similar meanings are close in vector space!

distance("stock", "share") < distance("stock", "banana")


HOW ARE THEY LEARNED?
---------------------
From large text corpora, by predicting:
- Context from word (Skip-gram)
- Word from context (CBOW)

"You shall know a word by the company it keeps"
  - J.R. Firth, 1957


APPLICATIONS:
-------------
- Document classification
- Sentiment analysis
- Named entity recognition
- Machine translation
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Word Embedding Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: 2D visualization of word embeddings
ax2 = axes[0, 1]

# Simulated 2D positions for word clusters
np.random.seed(42)

# Finance words
finance_words = ['stock', 'bond', 'equity', 'market', 'trading']
finance_x = np.random.randn(5) * 0.3 + 2
finance_y = np.random.randn(5) * 0.3 + 2

# Tech words
tech_words = ['computer', 'software', 'algorithm', 'data', 'network']
tech_x = np.random.randn(5) * 0.3 - 2
tech_y = np.random.randn(5) * 0.3 + 1

# Food words
food_words = ['apple', 'banana', 'orange', 'fruit', 'food']
food_x = np.random.randn(5) * 0.3 + 0
food_y = np.random.randn(5) * 0.3 - 2

ax2.scatter(finance_x, finance_y, c=MLGREEN, s=100, label='Finance')
ax2.scatter(tech_x, tech_y, c=MLBLUE, s=100, label='Tech')
ax2.scatter(food_x, food_y, c=MLORANGE, s=100, label='Food')

# Label words
for word, x, y in zip(finance_words, finance_x, finance_y):
    ax2.annotate(word, (x, y), fontsize=8, ha='center', va='bottom')
for word, x, y in zip(tech_words, tech_x, tech_y):
    ax2.annotate(word, (x, y), fontsize=8, ha='center', va='bottom')
for word, x, y in zip(food_words, food_x, food_y):
    ax2.annotate(word, (x, y), fontsize=8, ha='center', va='bottom')

ax2.set_xlabel('Dimension 1')
ax2.set_ylabel('Dimension 2')
ax2.set_title('Word Embeddings in 2D (t-SNE)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: One-hot vs embedding comparison
ax3 = axes[1, 0]
ax3.axis('off')

comparison = '''
ONE-HOT vs EMBEDDING COMPARISON

ONE-HOT ENCODING:
-----------------
Vocabulary: [apple, banana, stock, bond, ...]

"apple"  = [1, 0, 0, 0, 0, ..., 0]
"banana" = [0, 1, 0, 0, 0, ..., 0]
"stock"  = [0, 0, 1, 0, 0, ..., 0]

Properties:
- Dimension = vocabulary size (huge!)
- All words are orthogonal
- No semantic similarity
- Dot product: apple . banana = 0


WORD EMBEDDINGS:
----------------
"apple"  = [0.21, -0.45, 0.87, 0.12, ...]  (300-dim)
"banana" = [0.19, -0.42, 0.89, 0.15, ...]  (similar!)
"stock"  = [-0.32, 0.76, -0.21, 0.54, ...]

Properties:
- Dimension = 100-300 (small, fixed)
- Similar words have similar vectors
- Captures semantic relationships
- Dot product: apple . banana = 0.95


SEMANTIC PROPERTIES:
--------------------
king - man + woman = queen
paris - france + italy = rome
stock + increase = growth


WHY EMBEDDINGS WIN:
-------------------
1. Lower dimensionality = faster models
2. Semantic similarity = better generalization
3. Transfer learning = use pretrained embeddings
'''

ax3.text(0.02, 0.98, comparison, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('One-Hot vs Embedding', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Dimensionality comparison
ax4 = axes[1, 1]

methods = ['One-Hot\n(50K vocab)', 'TF-IDF\n(5K features)', 'Word2Vec\n(300 dim)', 'FastText\n(300 dim)']
dimensions = [50000, 5000, 300, 300]
semantic_info = [0, 0.3, 0.9, 0.95]

x = np.arange(len(methods))
width = 0.35

bars1 = ax4.bar(x - width/2, [d/1000 for d in dimensions], width, label='Dimensions (K)', color=MLBLUE, edgecolor='black')
ax4_twin = ax4.twinx()
bars2 = ax4_twin.bar(x + width/2, semantic_info, width, label='Semantic Info', color=MLGREEN, edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(methods, fontsize=8)
ax4.set_ylabel('Dimensions (thousands)', color=MLBLUE)
ax4_twin.set_ylabel('Semantic Information', color=MLGREEN)
ax4.set_title('Comparison of Text Representations', fontsize=11, fontweight='bold', color=MLPURPLE)

ax4.set_ylim(0, 60)
ax4_twin.set_ylim(0, 1.1)

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
