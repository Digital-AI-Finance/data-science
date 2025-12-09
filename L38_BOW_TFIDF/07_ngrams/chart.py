"""N-grams - Word Sequences as Features"""
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
fig.suptitle('N-grams: Capturing Word Sequences', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: N-gram concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT ARE N-GRAMS?

DEFINITION:
-----------
N-grams are sequences of N consecutive words.
They capture local word order and phrases.


TYPES:
------
Unigrams (n=1): Single words
  "stock", "price", "rose"

Bigrams (n=2): Two-word sequences
  "stock price", "price rose"

Trigrams (n=3): Three-word sequences
  "stock price rose"


EXAMPLE:
--------
Text: "stock price rose sharply"

Unigrams: ["stock", "price", "rose", "sharply"]
Bigrams:  ["stock price", "price rose", "rose sharply"]
Trigrams: ["stock price rose", "price rose sharply"]


WHY USE N-GRAMS?
----------------
1. Capture phrases: "interest rate" is different from "interest" + "rate"
2. Handle negation: "not good" vs just "good"
3. Better for sentiment: "highly recommended" vs "not recommended"
4. Domain terms: "P/E ratio", "earnings per share"


COST:
-----
More n-grams = much larger vocabulary!
"the" + "cat" + "sat" = 3 unigrams
But: "the cat", "cat sat" = 2 more bigrams
And: "the cat sat" = 1 more trigram

Vocabulary grows exponentially with n.
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('N-gram Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual n-gram extraction
ax2 = axes[0, 1]
ax2.axis('off')

sentence = "Apple stock rose today"
words = sentence.split()

# Draw sentence
ax2.text(0.5, 0.95, f'"{sentence}"', fontsize=12, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=MLBLUE, alpha=0.3))

# Draw word boxes
x_positions = [0.15, 0.35, 0.55, 0.75]
for i, (x, word) in enumerate(zip(x_positions, words)):
    ax2.text(x, 0.75, word, fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.6))

# Unigrams
ax2.text(0.05, 0.55, 'Unigrams:', fontsize=9, fontweight='bold')
for i, word in enumerate(words):
    ax2.text(0.2 + i*0.2, 0.55, word, fontsize=8, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLGREEN, alpha=0.5))

# Bigrams
bigrams = ["Apple stock", "stock rose", "rose today"]
ax2.text(0.05, 0.35, 'Bigrams:', fontsize=9, fontweight='bold')
for i, bigram in enumerate(bigrams):
    ax2.text(0.25 + i*0.25, 0.35, bigram, fontsize=8, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLORANGE, alpha=0.5))

# Trigrams
trigrams = ["Apple stock rose", "stock rose today"]
ax2.text(0.05, 0.15, 'Trigrams:', fontsize=9, fontweight='bold')
for i, trigram in enumerate(trigrams):
    ax2.text(0.3 + i*0.35, 0.15, trigram, fontsize=8, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLRED, alpha=0.4))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('N-gram Extraction', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: sklearn implementation
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
N-GRAMS IN SKLEARN

from sklearn.feature_extraction.text import TfidfVectorizer


# UNIGRAMS ONLY (default)
vectorizer = TfidfVectorizer(ngram_range=(1, 1))


# BIGRAMS ONLY
vectorizer = TfidfVectorizer(ngram_range=(2, 2))


# UNIGRAMS + BIGRAMS (most common)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))


# UNIGRAMS + BIGRAMS + TRIGRAMS
vectorizer = TfidfVectorizer(ngram_range=(1, 3))


# EXAMPLE
docs = [
    "stock price rose",
    "stock price fell"
]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
# ['fell', 'price', 'price fell', 'price rose',
#  'rose', 'stock', 'stock price']


# CONTROL VOCABULARY EXPLOSION
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,    # Limit total features
    min_df=2              # Bigrams must appear in 2+ docs
)


# CHECK VOCABULARY SIZE
print(f"Vocab size: {len(vectorizer.vocabulary_)}")


# FINANCE-SPECIFIC BIGRAMS
# "interest rate", "earnings per", "market cap"
# These are captured automatically with bigrams!
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Vocabulary size comparison
ax4 = axes[1, 1]

# Simulated vocabulary sizes
ngram_configs = ['(1,1)\nUnigrams', '(1,2)\n+Bigrams', '(1,3)\n+Trigrams', '(2,3)\nBi+Tri']
vocab_sizes = [10000, 45000, 120000, 110000]
accuracies = [0.82, 0.87, 0.88, 0.84]

x = np.arange(len(ngram_configs))
width = 0.35

bars = ax4.bar(x, vocab_sizes, width, color=MLBLUE, edgecolor='black', label='Vocab Size')
ax4_twin = ax4.twinx()
line = ax4_twin.plot(x, accuracies, 'o-', color=MLGREEN, linewidth=2, markersize=10, label='Accuracy')

ax4.set_xticks(x)
ax4.set_xticklabels(ngram_configs)
ax4.set_ylabel('Vocabulary Size', color=MLBLUE)
ax4_twin.set_ylabel('Accuracy', color=MLGREEN)
ax4.set_title('N-gram Range vs Vocab Size & Accuracy', fontsize=11, fontweight='bold', color=MLPURPLE)

# Highlight recommended
ax4.bar(1, vocab_sizes[1], width, color=MLGREEN, edgecolor='black', alpha=0.5)
ax4.text(1, vocab_sizes[1] + 5000, 'Recommended', ha='center', fontsize=8, color=MLGREEN, fontweight='bold')

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
