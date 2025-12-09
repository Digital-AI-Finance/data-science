"""Stemming and Lemmatization - Normalizing Words"""
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
fig.suptitle('Stemming and Lemmatization: Normalizing Words', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Concept explanation
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
STEMMING vs LEMMATIZATION

GOAL: Reduce words to their base/root form


STEMMING:
---------
- Chops off word endings (crude)
- Fast but imprecise
- May produce non-words

Examples:
  running  -> run
  studies  -> studi (not a word!)
  better   -> better (unchanged)
  earnings -> earn


LEMMATIZATION:
--------------
- Uses vocabulary + grammar
- Returns actual words (lemmas)
- Slower but accurate

Examples:
  running  -> run
  studies  -> study
  better   -> good (knows it's comparative)
  earnings -> earning


WHY DO THIS?
------------
1. Reduces vocabulary size
2. Groups related words together
3. Improves search/matching
4. Better for bag-of-words models


COMPARISON:
-----------
Word        | Stem    | Lemma
------------|---------|-------
running     | run     | run
ran         | ran     | run
better      | better  | good
studies     | studi   | study
earnings    | earn    | earning
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Concept Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual comparison
ax2 = axes[0, 1]
ax2.axis('off')

# Words and their transformations
words = ['trading', 'traded', 'trades', 'trader']
stems = ['trade', 'trade', 'trade', 'trader']  # Some errors
lemmas = ['trade', 'trade', 'trade', 'trader']

# Draw flow diagram
y_positions = [0.8, 0.6, 0.4, 0.2]

for i, (word, stem, lemma) in enumerate(zip(words, stems, lemmas)):
    y = y_positions[i]

    # Original word
    ax2.text(0.15, y, word, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLBLUE, alpha=0.5))

    # Arrows
    ax2.annotate('', xy=(0.38, y), xytext=(0.28, y),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=1.5))
    ax2.annotate('', xy=(0.72, y), xytext=(0.62, y),
                arrowprops=dict(arrowstyle='->', color=MLGREEN, lw=1.5))

    # Stem
    ax2.text(0.5, y, stem, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLORANGE, alpha=0.5))

    # Lemma
    ax2.text(0.85, y, lemma, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLGREEN, alpha=0.5))

# Headers
ax2.text(0.15, 0.95, 'Original', fontsize=10, ha='center', fontweight='bold', color=MLBLUE)
ax2.text(0.5, 0.95, 'Stemmed', fontsize=10, ha='center', fontweight='bold', color=MLORANGE)
ax2.text(0.85, 0.95, 'Lemmatized', fontsize=10, ha='center', fontweight='bold', color=MLGREEN)

ax2.text(0.5, 0.05, 'All variations map to same base form', fontsize=10, ha='center', style='italic')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Word Normalization Flow', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Python code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
STEMMING AND LEMMATIZATION IN PYTHON

# STEMMING with NLTK
from nltk.stem import PorterStemmer, SnowballStemmer

# Porter Stemmer (most common)
porter = PorterStemmer()
words = ['trading', 'traded', 'trades', 'trader', 'earnings']

stems = [porter.stem(w) for w in words]
print(stems)  # ['trade', 'trade', 'trade', 'trader', 'earn']


# Snowball Stemmer (improved)
snowball = SnowballStemmer('english')
stems = [snowball.stem(w) for w in words]


# LEMMATIZATION with NLTK
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Default: treats word as noun
print(lemmatizer.lemmatize('running'))  # 'running' (noun)
print(lemmatizer.lemmatize('running', pos='v'))  # 'run' (verb)


# LEMMATIZATION with spaCy (recommended)
import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp("The traders were trading stocks actively")
lemmas = [token.lemma_ for token in doc]
print(lemmas)  # ['the', 'trader', 'be', 'trade', 'stock', 'actively']


# COMPLETE PIPELINE
def normalize_text(text, use_lemma=True):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text.lower())

    if use_lemma:
        return [token.lemma_ for token in doc if token.is_alpha]
    else:
        stemmer = PorterStemmer()
        return [stemmer.stem(token.text) for token in doc if token.is_alpha]
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: When to use which
ax4 = axes[1, 1]
ax4.axis('off')

usage = '''
WHEN TO USE WHICH?

USE STEMMING WHEN:
------------------
- Speed is critical
- Large datasets
- Simple applications
- Don't need real words
- Information retrieval


USE LEMMATIZATION WHEN:
-----------------------
- Accuracy matters
- Need actual words
- Sentiment analysis
- Text generation
- Small/medium datasets


NEITHER NEEDED FOR:
-------------------
- Deep learning with embeddings
- Modern transformers (BERT, GPT)
- Subword tokenization


FINANCE-SPECIFIC CONSIDERATIONS:
--------------------------------
Stock terms:
  "IPOs" -> "IPO" (lemma)
  "IPOs" -> "ipo" (stem, loses case)

Financial verbs:
  "outperforming" -> "outperform" (lemma)
  "outperforming" -> "outperform" (stem, same!)

Tickers - DON'T NORMALIZE:
  "AAPL" should stay "AAPL"
  Never lowercase or stem tickers!


BEST PRACTICE:
--------------
1. Try lemmatization first
2. If too slow, use stemming
3. For deep learning, skip both
4. Always preserve special tokens (tickers, $, %)


VOCABULARY REDUCTION:
---------------------
Original vocab:  50,000 words
After stemming:  ~35,000 (30% reduction)
After lemmatization: ~38,000 (24% reduction)
'''

ax4.text(0.02, 0.98, usage, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('When to Use Which?', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
