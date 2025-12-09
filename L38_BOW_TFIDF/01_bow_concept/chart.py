"""Bag of Words Concept"""
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
fig.suptitle('Bag of Words: Turning Text into Numbers', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Concept explanation
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
BAG OF WORDS (BoW)

WHAT IS IT?
-----------
A way to represent text as numbers.
Count how many times each word appears.
Ignore word order (hence "bag").


THE IDEA:
---------
1. Create vocabulary (all unique words)
2. For each document, count word occurrences
3. Result: document = vector of counts


EXAMPLE:
--------
Doc 1: "stock price rose"
Doc 2: "stock price fell"

Vocabulary: [stock, price, rose, fell]

Doc 1 vector: [1, 1, 1, 0]
Doc 2 vector: [1, 1, 0, 1]


WHY USE IT?
-----------
- Simple and effective
- Works with any ML algorithm
- Fast to compute
- Good baseline


LIMITATIONS:
------------
- Loses word order
- "Not good" vs "Good" treated similarly
- Large vocabularies = high dimensions
- Rare words and common words treated equally


NEXT STEP: TF-IDF
-----------------
Weights words by importance, not just counts.
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Bag of Words Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual example
ax2 = axes[0, 1]
ax2.axis('off')

# Document examples
docs = [
    "earnings beat expectations",
    "earnings missed estimates",
    "stock rose on earnings"
]

# Build vocabulary
all_words = ['earnings', 'beat', 'expectations', 'missed', 'estimates', 'stock', 'rose', 'on']

# BoW matrix
bow_matrix = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0],  # doc 1
    [1, 0, 0, 1, 1, 0, 0, 0],  # doc 2
    [1, 0, 0, 0, 0, 1, 1, 1],  # doc 3
])

# Draw document transformation
ax2.text(0.5, 0.95, 'Documents to Vectors', fontsize=12, ha='center', fontweight='bold', color=MLPURPLE)

for i, doc in enumerate(docs):
    y = 0.8 - i * 0.25
    # Document text
    ax2.text(0.25, y, f'"{doc}"', fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLBLUE, alpha=0.4))
    # Arrow
    ax2.annotate('', xy=(0.5, y), xytext=(0.4, y),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
    # Vector
    vector_str = str(list(bow_matrix[i]))
    ax2.text(0.75, y, vector_str, fontsize=8, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLGREEN, alpha=0.4))

# Vocabulary
ax2.text(0.5, 0.1, f'Vocabulary: {all_words[:4]}...', fontsize=8, ha='center',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.6))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Transformation Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: BoW matrix visualization
ax3 = axes[1, 0]

# Create heatmap
vocab_short = ['earn.', 'beat', 'expect.', 'miss', 'estim.', 'stock', 'rose']
bow_short = bow_matrix[:, :7]

im = ax3.imshow(bow_short, cmap='Blues', aspect='auto')

ax3.set_xticks(range(len(vocab_short)))
ax3.set_xticklabels(vocab_short, rotation=45, ha='right', fontsize=8)
ax3.set_yticks(range(3))
ax3.set_yticklabels(['Doc 1', 'Doc 2', 'Doc 3'])

# Add count labels
for i in range(bow_short.shape[0]):
    for j in range(bow_short.shape[1]):
        ax3.text(j, i, bow_short[i, j], ha='center', va='center',
                color='white' if bow_short[i, j] > 0 else 'black', fontsize=10)

ax3.set_title('BoW Matrix (Document-Term)', fontsize=11, fontweight='bold', color=MLPURPLE)
plt.colorbar(im, ax=ax3, label='Word Count')

# Plot 4: Key properties
ax4 = axes[1, 1]
ax4.axis('off')

properties = '''
KEY PROPERTIES OF BOW

MATRIX DIMENSIONS:
------------------
Rows: Number of documents (n)
Columns: Vocabulary size (V)

Result: n x V matrix

Financial news corpus example:
- 10,000 articles
- 50,000 unique words
- Matrix: 10,000 x 50,000 = 500 million cells!


SPARSE REPRESENTATION:
----------------------
Most cells are 0 (word not in document).
Use sparse matrices to save memory.


DOCUMENT SIMILARITY:
--------------------
Can compare documents using their vectors!

Cosine similarity:
sim(doc1, doc2) = (v1 . v2) / (|v1| * |v2|)


USED FOR:
---------
- Text classification
- Document clustering
- Information retrieval
- Sentiment analysis


SKLEARN USAGE:
--------------
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# X is now a sparse matrix!
print(X.shape)  # (n_docs, vocab_size)
'''

ax4.text(0.02, 0.98, properties, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Key Properties', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
