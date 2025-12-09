"""Similarity Search with Word Embeddings"""
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
fig.suptitle('Similarity Search with Word Embeddings', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Similarity metrics
ax1 = axes[0, 0]
ax1.axis('off')

metrics = '''
SIMILARITY METRICS

COSINE SIMILARITY (most common):
--------------------------------
cos(a, b) = (a . b) / (|a| * |b|)

Range: -1 to 1
1 = identical direction
0 = orthogonal
-1 = opposite direction

Example:
a = [0.5, 0.5, 0.7]
b = [0.4, 0.6, 0.65]
cos(a, b) = 0.98 (very similar!)


EUCLIDEAN DISTANCE:
-------------------
dist(a, b) = sqrt(sum((a_i - b_i)^2))

Range: 0 to infinity
0 = identical
Larger = more different

Note: Sensitive to vector magnitude.


DOT PRODUCT:
------------
dot(a, b) = sum(a_i * b_i)

Fast to compute.
Not normalized (depends on magnitudes).


WHICH TO USE?
-------------
- Cosine: Best for word embeddings (normalized)
- Euclidean: Good for general vectors
- Dot product: Fast, used internally


IN CODE:
--------
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cosine

cos_sim = cosine_similarity([a], [b])[0][0]
euc_dist = euclidean(a, b)
cos_dist = cosine(a, b)  # 1 - cosine_similarity
'''

ax1.text(0.02, 0.98, metrics, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Similarity Metrics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual similarity example
ax2 = axes[0, 1]

# Simulated similarity scores
query_word = "stock"
similar_words = ['share', 'equity', 'bond', 'market', 'price', 'apple', 'computer', 'banana']
similarities = [0.92, 0.88, 0.75, 0.72, 0.68, 0.35, 0.25, 0.08]

colors = [MLGREEN if s > 0.6 else MLORANGE if s > 0.3 else MLRED for s in similarities]

bars = ax2.barh(range(len(similar_words)), similarities, color=colors, edgecolor='black')
ax2.set_yticks(range(len(similar_words)))
ax2.set_yticklabels(similar_words)
ax2.set_xlabel('Cosine Similarity')
ax2.set_title(f'Words Similar to "{query_word}"', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.axvline(0.6, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlim(0, 1)
ax2.invert_yaxis()
ax2.grid(alpha=0.3, axis='x')

# Add threshold annotation
ax2.text(0.62, 7, 'Threshold', fontsize=8, color='gray')

# Plot 3: Finding similar documents
ax3 = axes[1, 0]
ax3.axis('off')

doc_search = '''
DOCUMENT SIMILARITY WITH EMBEDDINGS

# STEP 1: Get document vector (average of word vectors)
def doc_to_vector(text, model):
    words = text.lower().split()
    vectors = [model.wv[w] for w in words if w in model.wv]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


# STEP 2: Compute all document vectors
doc_vectors = np.array([doc_to_vector(doc, model) for doc in documents])


# STEP 3: Find similar documents
from sklearn.metrics.pairwise import cosine_similarity

query = "tech stock earnings report"
query_vec = doc_to_vector(query, model)

# Compute similarities
similarities = cosine_similarity([query_vec], doc_vectors)[0]

# Get top-k most similar
top_k = 5
top_indices = similarities.argsort()[-top_k:][::-1]

print("Most similar documents:")
for idx in top_indices:
    print(f"  {similarities[idx]:.3f}: {documents[idx][:50]}...")


# OPTIMIZED FOR LARGE COLLECTIONS
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, metric='cosine')
nn.fit(doc_vectors)

distances, indices = nn.kneighbors([query_vec])
'''

ax3.text(0.02, 0.98, doc_search, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Document Similarity', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Use cases
ax4 = axes[1, 1]
ax4.axis('off')

use_cases = '''
SIMILARITY SEARCH USE CASES

1. FIND SIMILAR NEWS ARTICLES:
------------------------------
Query: "Fed raises interest rates"
Find: Similar articles about monetary policy

Application: News recommendation, clustering


2. FIND RELATED STOCKS:
-----------------------
Query: Company description of AAPL
Find: Companies with similar business models

Application: Competitor analysis, sector grouping


3. SEARCH EARNINGS CALLS:
-------------------------
Query: "supply chain disruptions"
Find: Earnings calls mentioning similar issues

Application: Risk analysis, theme tracking


4. SEMANTIC SEARCH:
-------------------
Query: "company losing money" (informal)
Find: Documents about "operating losses" (formal)

Application: Better search than keyword matching!


5. DUPLICATE DETECTION:
-----------------------
Find: Documents that are near-duplicates
Threshold: cosine_similarity > 0.95

Application: Data cleaning, deduplication


PERFORMANCE TIPS:
-----------------
- Precompute document vectors
- Use approximate nearest neighbors for large datasets
  (FAISS, Annoy, ScaNN)
- Normalize vectors for faster cosine computation
- Batch process queries
'''

ax4.text(0.02, 0.98, use_cases, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Use Cases', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
