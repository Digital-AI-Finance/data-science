"""Word2Vec - Learning Word Representations"""
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
fig.suptitle('Word2Vec: Learning Word Representations', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Word2Vec concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WORD2VEC (Mikolov et al., 2013)

CORE IDEA:
----------
Learn word vectors by predicting words from context.

"You shall know a word by the company it keeps"


TWO ARCHITECTURES:
------------------

1. SKIP-GRAM:
   Given a word, predict its context words.

   Input: "stock"
   Output: ["the", "price", "rose", "today"]

   Better for rare words, larger datasets.


2. CBOW (Continuous Bag of Words):
   Given context words, predict the target word.

   Input: ["the", "price", "rose", "today"]
   Output: "stock"

   Faster training, good for frequent words.


TRAINING OBJECTIVE:
-------------------
Maximize probability of context given word (Skip-gram)
or word given context (CBOW).

Uses negative sampling for efficiency.


KEY PARAMETERS:
---------------
- vector_size: Dimension of embeddings (100-300)
- window: Context window size (5-10)
- min_count: Ignore words with freq < this
- epochs: Training iterations (5-20)


RESULT:
-------
A matrix of shape: (vocabulary_size, vector_size)
Each row is a word's embedding vector.
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Word2Vec Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Skip-gram visualization
ax2 = axes[0, 1]
ax2.axis('off')

# Draw Skip-gram architecture
sentence = "the stock price rose today"
words = sentence.split()
center_idx = 2  # "price" is center

# Draw words
y = 0.8
for i, word in enumerate(words):
    x = 0.1 + i * 0.18
    color = MLRED if i == center_idx else MLBLUE
    ax2.text(x, y, word, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

# Draw center word connection
center_x = 0.1 + center_idx * 0.18
ax2.text(center_x, 0.5, 'INPUT\n"price"', fontsize=10, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=MLRED, alpha=0.3))

# Draw context predictions
context = ['stock', 'rose']
for i, ctx in enumerate(context):
    x = 0.25 + i * 0.5
    ax2.text(x, 0.2, f'PREDICT\n"{ctx}"', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLGREEN, alpha=0.3))
    ax2.annotate('', xy=(x, 0.3), xytext=(center_x, 0.42),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))

ax2.text(0.5, 0.95, 'Skip-gram: Predict context from word', fontsize=11, ha='center',
         fontweight='bold', color=MLPURPLE)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Skip-gram Architecture', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Training intuition
ax3 = axes[1, 0]
ax3.axis('off')

training = '''
WORD2VEC TRAINING INTUITION

STARTING POINT:
---------------
Random vectors for all words.
"stock" = [0.12, -0.34, 0.56, ...]  (random)


TRAINING EXAMPLES:
------------------
From: "The stock price rose sharply"

Skip-gram examples (window=2):
("stock", "the")
("stock", "price")
("price", "stock")
("price", "rose")
...

The model learns:
- If "stock" appears near "price" often,
  their vectors should be similar.


AFTER TRAINING:
---------------
Words in similar contexts -> similar vectors

"stock" is often near: price, market, share, trading
"bond" is often near: price, market, yield, interest

So "stock" and "bond" end up with similar vectors!


WHAT THE VECTORS CAPTURE:
-------------------------
- Semantic similarity: stock ~ equity ~ share
- Syntactic patterns: running : ran :: jumping : jumped
- Analogies: king - man + woman = queen


NEGATIVE SAMPLING:
------------------
Instead of predicting all vocab words,
sample a few "negative" (random) words.

Makes training much faster!
'''

ax3.text(0.02, 0.98, training, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Training Intuition', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Embedding dimensions effect
ax4 = axes[1, 1]

dimensions = [50, 100, 200, 300, 500]
accuracy = [0.72, 0.78, 0.83, 0.85, 0.85]
training_time = [1, 2, 4, 6, 10]

ax4.plot(dimensions, accuracy, 'o-', color=MLBLUE, linewidth=2, markersize=8, label='Accuracy')

ax4_twin = ax4.twinx()
ax4_twin.plot(dimensions, training_time, 's--', color=MLRED, linewidth=2, markersize=8, label='Training Time')

ax4.set_xlabel('Embedding Dimensions')
ax4.set_ylabel('Task Accuracy', color=MLBLUE)
ax4_twin.set_ylabel('Training Time (hours)', color=MLRED)
ax4.set_title('Effect of Embedding Dimensions', fontsize=11, fontweight='bold', color=MLPURPLE)

# Mark common choice
ax4.axvline(300, color=MLGREEN, linestyle='--', alpha=0.7)
ax4.text(310, 0.74, 'Common\nchoice', fontsize=9, color=MLGREEN)

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
