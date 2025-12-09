"""Gensim - Word2Vec in Python"""
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
fig.suptitle('Gensim: Word2Vec in Python', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic usage
ax1 = axes[0, 0]
ax1.axis('off')

basic = '''
GENSIM WORD2VEC BASICS

# Installation
pip install gensim


# Import
from gensim.models import Word2Vec


# Prepare data (list of tokenized sentences)
sentences = [
    ['stock', 'price', 'rose', 'today'],
    ['market', 'rallied', 'on', 'earnings'],
    ['tech', 'stocks', 'outperformed'],
    ['bond', 'yields', 'fell', 'sharply'],
    # ... more sentences
]


# Train model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,    # Embedding dimensions
    window=5,           # Context window size
    min_count=2,        # Ignore rare words
    workers=4,          # Parallel threads
    epochs=10           # Training iterations
)


# Access word vector
vector = model.wv['stock']
print(vector.shape)  # (100,)


# Save and load
model.save('word2vec.model')
model = Word2Vec.load('word2vec.model')


# Check vocabulary
print(len(model.wv))  # Vocabulary size
print('stock' in model.wv)  # Check if word exists
'''

ax1.text(0.02, 0.98, basic, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Basic Usage', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Finding similar words
ax2 = axes[0, 1]
ax2.axis('off')

similarity = '''
FINDING SIMILAR WORDS

# Most similar words
similar = model.wv.most_similar('stock', topn=5)
print(similar)
# [('share', 0.89), ('equity', 0.85), ('bond', 0.82), ...]


# Similarity between two words
sim = model.wv.similarity('stock', 'bond')
print(f"Similarity: {sim:.3f}")  # ~0.75


# Words similar to multiple words
similar = model.wv.most_similar(
    positive=['stock', 'technology'],
    topn=5
)
# Tech stocks!


# Find odd one out
odd = model.wv.doesnt_match(['stock', 'bond', 'equity', 'banana'])
print(f"Odd one out: {odd}")  # 'banana'


# Word analogies
# king - man + woman = ?
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
print(result)  # [('queen', 0.85)]


# Finance example: stock - company + economy = ?
result = model.wv.most_similar(
    positive=['stock', 'economy'],
    negative=['company'],
    topn=3
)
# Might get: market, index, gdp


# Distance (opposite of similarity)
distance = model.wv.distance('stock', 'banana')
print(f"Distance: {distance:.3f}")
'''

ax2.text(0.02, 0.98, similarity, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Finding Similar Words', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Training parameters
ax3 = axes[1, 0]
ax3.axis('off')

params = '''
TRAINING PARAMETERS

from gensim.models import Word2Vec


model = Word2Vec(
    sentences,           # List of tokenized sentences

    # ARCHITECTURE
    vector_size=100,     # Dimension of word vectors
                         # 100-300 common, 300 for rich semantics

    window=5,            # Context window (words before/after)
                         # 5-10 typical, larger = more topical

    # TRAINING
    min_count=5,         # Ignore words with freq < this
                         # Removes rare/noisy words

    epochs=10,           # Training passes over data
                         # More = better, but diminishing returns

    # ALGORITHM
    sg=0,                # 0=CBOW, 1=Skip-gram
                         # Skip-gram better for rare words

    negative=5,          # Negative samples per word
                         # 5-20 typical

    # PERFORMANCE
    workers=4,           # Parallel threads
)


RECOMMENDED SETTINGS:
---------------------
Small corpus (<1M words):
  vector_size=100, window=5, min_count=2

Large corpus (>10M words):
  vector_size=300, window=10, min_count=5

Financial text:
  vector_size=200, window=8, min_count=3, sg=1


TRAINING TIME ESTIMATE:
-----------------------
1M words: ~5 minutes
10M words: ~30 minutes
100M words: ~4 hours
(with 4 workers)
'''

ax3.text(0.02, 0.98, params, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Training Parameters', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete example
ax4 = axes[1, 1]
ax4.axis('off')

example = '''
COMPLETE FINANCIAL TEXT EXAMPLE

from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize


# 1. PREPARE DATA
documents = [
    "Apple stock rose on strong earnings report",
    "Tech stocks rallied amid market optimism",
    "Bond yields fell as investors sought safety",
    "Market volatility increased on Fed comments",
    # ... many more documents
]

# Tokenize
sentences = [word_tokenize(doc.lower()) for doc in documents]


# 2. TRAIN MODEL
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    epochs=20,
    sg=1  # Skip-gram
)


# 3. USE MODEL
# Similar to "stock"
print(model.wv.most_similar('stock', topn=3))

# Get vector for ML
def get_doc_vector(tokens, model):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

doc_vectors = [get_doc_vector(s, model) for s in sentences]


# 4. SAVE
model.save('finance_word2vec.model')


# 5. LOAD LATER
model = Word2Vec.load('finance_word2vec.model')
'''

ax4.text(0.02, 0.98, example, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
