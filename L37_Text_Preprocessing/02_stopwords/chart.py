"""Stopwords - Removing Common Words"""
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
fig.suptitle('Stopwords: Removing Common Words', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What are stopwords
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT ARE STOPWORDS?

DEFINITION:
-----------
Common words that carry little meaning on their own.
They're "noise" for many NLP tasks.


EXAMPLES:
---------
English: the, a, an, is, are, was, were, be, been,
         have, has, do, does, did, will, would, could,
         should, may, might, can, to, of, in, for, on,
         with, at, by, from, as, it, this, that, which


WHY REMOVE THEM?
----------------
1. They're very frequent (don't distinguish documents)
2. Add noise to analysis
3. Reduce vocabulary size
4. Speed up processing


EXAMPLE:
--------
Original: "The stock market is going to rise in the future"
Cleaned:  "stock market going rise future"

Original: "Apple is a company that makes technology products"
Cleaned:  "Apple company makes technology products"


WHEN TO KEEP STOPWORDS:
-----------------------
- Sentiment analysis: "not good" vs "good"
- Question answering: "what", "when", "where"
- Machine translation
- Language modeling


WHEN TO REMOVE:
---------------
- Topic modeling
- Document classification
- Keyword extraction
- Information retrieval
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What Are Stopwords?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Stopword frequency visualization
ax2 = axes[0, 1]

# Common words and their frequencies (simulated)
words = ['the', 'is', 'at', 'stock', 'a', 'market', 'to', 'price', 'of', 'earnings',
         'and', 'revenue', 'in', 'growth', 'for', 'profit']
frequencies = [150, 120, 100, 85, 95, 75, 90, 70, 88, 60, 82, 55, 78, 50, 72, 45]
is_stopword = [True, True, True, False, True, False, True, False, True, False,
               True, False, True, False, True, False]

colors = [MLRED if sw else MLGREEN for sw in is_stopword]

bars = ax2.barh(range(len(words)), frequencies, color=colors, edgecolor='black')
ax2.set_yticks(range(len(words)))
ax2.set_yticklabels(words)
ax2.set_xlabel('Frequency')
ax2.set_title('Word Frequencies in Financial Text', fontsize=11, fontweight='bold', color=MLPURPLE)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MLRED, label='Stopword (remove)'),
                   Patch(facecolor=MLGREEN, label='Content word (keep)')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)

ax2.invert_yaxis()
ax2.grid(alpha=0.3, axis='x')

# Plot 3: Python code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
STOPWORD REMOVAL IN PYTHON

# METHOD 1: NLTK stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

text = "The stock market is going to rise in the future"
tokens = word_tokenize(text.lower())

# Remove stopwords
filtered = [w for w in tokens if w not in stop_words]
print(filtered)  # ['stock', 'market', 'going', 'rise', 'future']


# METHOD 2: Custom stopword list
custom_stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at'}

filtered = [w for w in tokens if w not in custom_stopwords]


# METHOD 3: spaCy (includes POS tagging)
import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp("The stock market is going to rise")
filtered = [token.text for token in doc if not token.is_stop]
# ['stock', 'market', 'going', 'rise']


# METHOD 4: Add custom stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Add finance-specific stopwords
stop_words.update(['said', 'according', 'also', 'would', 'could'])

# Remove finance terms you want to keep
stop_words.discard('not')  # Keep negations for sentiment


# CHECKING STOPWORDS:
print(f"Number of stopwords: {len(stop_words)}")
print(f"Is 'the' a stopword? {'the' in stop_words}")
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Before/after comparison
ax4 = axes[1, 1]
ax4.axis('off')

comparison = '''
BEFORE AND AFTER STOPWORD REMOVAL

EXAMPLE 1 - Earnings Report:
----------------------------
Before: "The company reported that its quarterly earnings
         were better than the analysts had expected"

After:  "company reported quarterly earnings better
         analysts expected"

Tokens: 14 -> 6 (57% reduction!)


EXAMPLE 2 - Market News:
------------------------
Before: "Stock prices in the technology sector are rising
         as investors are becoming more optimistic"

After:  "Stock prices technology sector rising
         investors becoming optimistic"

Tokens: 15 -> 8 (47% reduction!)


EXAMPLE 3 - Financial Analysis:
-------------------------------
Before: "The P/E ratio of this stock is higher than the
         industry average which suggests it may be overvalued"

After:  "P/E ratio stock higher industry average
         suggests overvalued"

Tokens: 18 -> 8 (56% reduction!)


CAUTION - SENTIMENT ANALYSIS:
-----------------------------
Before: "This is not a good investment"
After:  "good investment"  (Meaning reversed!)

Solution: Keep negations ['not', 'no', "n't", 'never']


FINANCE-SPECIFIC STOPWORDS TO ADD:
----------------------------------
['said', 'says', 'according', 'reported', 'announced',
 'would', 'could', 'may', 'might', 'year', 'quarter']
'''

ax4.text(0.02, 0.98, comparison, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Before/After Examples', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
