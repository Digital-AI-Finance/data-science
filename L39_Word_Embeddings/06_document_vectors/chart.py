"""Document Vectors - From Words to Documents"""
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
fig.suptitle('Document Vectors: From Words to Documents', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Methods overview
ax1 = axes[0, 0]
ax1.axis('off')

methods = '''
DOCUMENT VECTOR METHODS

THE CHALLENGE:
--------------
Word embeddings give us vectors for WORDS.
How do we get vectors for DOCUMENTS?


METHOD 1: SIMPLE AVERAGE
------------------------
doc_vec = mean(word_vectors)

Pros: Simple, fast, works well
Cons: Ignores word importance, order


METHOD 2: TF-IDF WEIGHTED AVERAGE
---------------------------------
doc_vec = sum(tfidf[word] * word_vec) / sum(tfidf)

Pros: Weights important words more
Cons: Needs TF-IDF computation


METHOD 3: DOC2VEC (Paragraph Vectors)
-------------------------------------
Learns document vectors directly (like Word2Vec).

Pros: Purpose-built, captures document themes
Cons: Needs training, more complex


METHOD 4: SENTENCE TRANSFORMERS
-------------------------------
Uses transformer models (BERT, etc.)

Pros: State-of-the-art quality
Cons: Slower, needs more resources


RECOMMENDATION:
---------------
Start: Simple average
Better: TF-IDF weighted
Best: Sentence transformers (if resources allow)
'''

ax1.text(0.02, 0.98, methods, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Methods Overview', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual aggregation
ax2 = axes[0, 1]
ax2.axis('off')

# Draw aggregation process
words = ['stock', 'price', 'rose', 'sharply']
y_positions = [0.75, 0.6, 0.45, 0.3]

# Word vectors
ax2.text(0.5, 0.95, '"Stock price rose sharply"', fontsize=11, ha='center',
         fontweight='bold', bbox=dict(facecolor=MLBLUE, alpha=0.3))

for word, y in zip(words, y_positions):
    ax2.text(0.2, y, word, fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.6))
    ax2.text(0.55, y, '[0.2, -0.3, 0.5, ...]', fontsize=8, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLGREEN, alpha=0.4))
    ax2.annotate('', xy=(0.35, y), xytext=(0.28, y),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=1.5))

# Aggregation arrow
ax2.annotate('', xy=(0.75, 0.5), xytext=(0.7, 0.5),
            arrowprops=dict(arrowstyle='->', color=MLRED, lw=3))
ax2.text(0.72, 0.55, 'Average', fontsize=9, color=MLRED)

# Document vector
ax2.text(0.88, 0.5, 'Doc\nVector', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor=MLRED, alpha=0.4))

ax2.set_xlim(0, 1)
ax2.set_ylim(0.15, 1)
ax2.set_title('Aggregation Process', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Implementation code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
DOCUMENT VECTOR IMPLEMENTATION

import numpy as np
from gensim.models import Word2Vec


# METHOD 1: SIMPLE AVERAGE
def doc_vector_average(tokens, model):
    """Average word vectors in document."""
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


# METHOD 2: TF-IDF WEIGHTED
def doc_vector_tfidf(tokens, model, tfidf_dict):
    """TF-IDF weighted average of word vectors."""
    vectors = []
    weights = []

    for token in tokens:
        if token in model.wv and token in tfidf_dict:
            vectors.append(model.wv[token])
            weights.append(tfidf_dict[token])

    if not vectors:
        return np.zeros(model.vector_size)

    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    return np.average(vectors, axis=0, weights=weights)


# METHOD 3: MAX POOLING
def doc_vector_max(tokens, model):
    """Max pooling over word vectors."""
    vectors = [model.wv[t] for t in tokens if t in model.wv]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.max(vectors, axis=0)


# CREATE DOCUMENT VECTORS FOR CORPUS
def create_doc_vectors(documents, model, method='average'):
    doc_vectors = []
    for doc in documents:
        tokens = doc.lower().split()
        if method == 'average':
            vec = doc_vector_average(tokens, model)
        elif method == 'max':
            vec = doc_vector_max(tokens, model)
        doc_vectors.append(vec)

    return np.array(doc_vectors)
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Method comparison
ax4 = axes[1, 1]

methods_names = ['Simple\nAverage', 'TF-IDF\nWeighted', 'Doc2Vec', 'Sentence\nTransformers']
accuracy = [0.75, 0.78, 0.80, 0.88]
speed = [100, 80, 50, 10]

x = np.arange(len(methods_names))
width = 0.35

bars1 = ax4.bar(x - width/2, accuracy, width, label='Accuracy', color=MLGREEN, edgecolor='black')
ax4_twin = ax4.twinx()
bars2 = ax4_twin.bar(x + width/2, speed, width, label='Speed (docs/sec)', color=MLBLUE, edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(methods_names, fontsize=8)
ax4.set_ylabel('Classification Accuracy', color=MLGREEN)
ax4_twin.set_ylabel('Processing Speed', color=MLBLUE)
ax4.set_title('Method Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_ylim(0.6, 1)

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
