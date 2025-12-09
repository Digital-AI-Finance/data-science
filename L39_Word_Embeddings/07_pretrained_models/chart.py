"""Pre-trained Word Embeddings"""
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
fig.suptitle('Pre-trained Word Embeddings', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Available models
ax1 = axes[0, 0]
ax1.axis('off')

models = '''
POPULAR PRE-TRAINED EMBEDDINGS

WORD2VEC (Google):
------------------
- Trained on: Google News (100B words)
- Dimensions: 300
- Vocabulary: 3 million words
- Size: ~3.5 GB


GLOVE (Stanford):
-----------------
- Trained on: Wikipedia + Gigaword
- Dimensions: 50, 100, 200, 300
- Vocabulary: 400K to 2.2M words
- Size: 100MB - 5GB


FASTTEXT (Facebook):
--------------------
- Trained on: Wikipedia + Common Crawl
- Dimensions: 300
- Vocabulary: 2 million words
- Size: ~7 GB
- Bonus: Handles out-of-vocabulary words!


WHICH TO USE?
-------------
- General text: GloVe or Word2Vec
- Need OOV handling: FastText
- Financial domain: Train your own!


DOWNLOAD LINKS:
---------------
- Word2Vec: https://code.google.com/archive/p/word2vec/
- GloVe: https://nlp.stanford.edu/projects/glove/
- FastText: https://fasttext.cc/docs/en/english-vectors.html
'''

ax1.text(0.02, 0.98, models, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Available Models', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Loading pre-trained
ax2 = axes[0, 1]
ax2.axis('off')

loading = '''
LOADING PRE-TRAINED EMBEDDINGS

# GENSIM (Word2Vec format)
from gensim.models import KeyedVectors

# Load Google's Word2Vec
model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin',
    binary=True
)

# Load GloVe (convert first)
# glove2word2vec(glove_file, word2vec_file)
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec('glove.6B.100d.txt', 'glove.word2vec.txt')
model = KeyedVectors.load_word2vec_format('glove.word2vec.txt')


# GENSIM (KeyedVectors for memory efficiency)
model = KeyedVectors.load_word2vec_format(
    'vectors.bin',
    binary=True,
    limit=100000  # Load only top 100K words
)


# FASTTEXT
import fasttext
model = fasttext.load_model('cc.en.300.bin')

# Or via gensim
from gensim.models import FastText
model = FastText.load_fasttext_format('cc.en.300.bin')


# USE THE MODEL
vector = model['stock']
similar = model.most_similar('stock', topn=5)


# MEMORY TIP
# For large models, use mmap for memory efficiency
model = KeyedVectors.load_word2vec_format(
    'vectors.bin',
    binary=True,
    mmap='r'  # Memory-mapped
)
'''

ax2.text(0.02, 0.98, loading, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Loading Pre-trained', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Model comparison
ax3 = axes[1, 0]

models_names = ['Word2Vec\n(Google)', 'GloVe\n(6B)', 'GloVe\n(840B)', 'FastText\n(Wiki)']
vocab_size = [3000, 400, 2200, 2000]  # in thousands
dimensions = [300, 300, 300, 300]
file_size = [3.5, 1.0, 5.0, 7.0]  # GB

x = np.arange(len(models_names))

ax3.bar(x, vocab_size, color=MLBLUE, edgecolor='black')
ax3.set_xticks(x)
ax3.set_xticklabels(models_names, fontsize=8)
ax3.set_ylabel('Vocabulary (thousands)')
ax3.set_title('Model Vocabulary Sizes', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.grid(alpha=0.3, axis='y')

# Add file size labels
for i, size in enumerate(file_size):
    ax3.text(i, vocab_size[i] + 50, f'{size}GB', ha='center', fontsize=8, color=MLRED)

# Plot 4: Train vs pretrained
ax4 = axes[1, 1]
ax4.axis('off')

comparison = '''
TRAIN YOUR OWN vs PRE-TRAINED

PRE-TRAINED ADVANTAGES:
-----------------------
+ No training needed
+ High quality (trained on billions of words)
+ General purpose
+ Works out of the box


TRAIN YOUR OWN ADVANTAGES:
--------------------------
+ Domain-specific vocabulary
+ Captures domain relationships
+ "Apple" = company, not fruit
+ Smaller, faster models


WHEN TO USE PRE-TRAINED:
------------------------
- General NLP tasks
- Small dataset (<1M words)
- Quick prototyping
- Limited compute resources


WHEN TO TRAIN YOUR OWN:
-----------------------
- Specialized domain (finance, medical, legal)
- Large domain corpus available
- Pre-trained performs poorly
- Need smaller model


HYBRID APPROACH:
----------------
1. Start with pre-trained
2. Fine-tune on domain data
3. Or: Combine both sets of vectors

from gensim.models import Word2Vec

# Load pre-trained
pretrained = KeyedVectors.load_word2vec_format('glove.bin')

# Train on domain data, initialize with pretrained
model = Word2Vec(vector_size=300, min_count=1)
model.build_vocab(domain_sentences)
model.wv.vectors = pretrained.vectors  # Initialize
model.train(domain_sentences, total_examples=len(domain_sentences), epochs=5)
'''

ax4.text(0.02, 0.98, comparison, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Train vs Pre-trained', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
