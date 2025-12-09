"""NLTK Basics - Natural Language Toolkit"""
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
fig.suptitle('NLTK: Natural Language Toolkit Basics', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is NLTK
ax1 = axes[0, 0]
ax1.axis('off')

intro = '''
WHAT IS NLTK?

DESCRIPTION:
------------
Natural Language Toolkit (NLTK)
- Most popular NLP library for Python
- Educational and research focused
- Rich set of tools and corpora


INSTALLATION:
-------------
pip install nltk

# Download required data
import nltk
nltk.download('punkt')       # Tokenizer
nltk.download('stopwords')   # Stopword list
nltk.download('wordnet')     # Lemmatizer
nltk.download('averaged_perceptron_tagger')  # POS tagger


KEY MODULES:
------------
nltk.tokenize    - Tokenization
nltk.corpus      - Text corpora and lexicons
nltk.stem        - Stemmers and lemmatizers
nltk.tag         - Part-of-speech tagging
nltk.chunk       - Named entity recognition
nltk.sentiment   - Sentiment analysis (VADER)


PROS:
-----
- Easy to learn
- Well documented
- Great for learning NLP
- Many built-in corpora

CONS:
-----
- Slower than spaCy
- Not ideal for production
- Some outdated algorithms
'''

ax1.text(0.02, 0.98, intro, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is NLTK?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: NLTK modules visualization
ax2 = axes[0, 1]
ax2.axis('off')

# Create a module diagram
modules = [
    ('nltk.tokenize', 'Split text into tokens'),
    ('nltk.corpus', 'Access text datasets'),
    ('nltk.stem', 'Stem/lemmatize words'),
    ('nltk.tag', 'Part-of-speech tags'),
    ('nltk.sentiment', 'Sentiment analysis'),
    ('nltk.chunk', 'Named entities')
]

y_positions = np.linspace(0.85, 0.15, len(modules))

for i, (module, desc) in enumerate(modules):
    y = y_positions[i]
    # Module box
    ax2.text(0.3, y, module, fontsize=10, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=MLBLUE, alpha=0.6))
    # Description
    ax2.text(0.7, y, desc, fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.6))
    # Arrow
    ax2.annotate('', xy=(0.48, y), xytext=(0.42, y),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=1.5))

ax2.text(0.5, 0.95, 'NLTK Core Modules', fontsize=11, ha='center', fontweight='bold', color=MLPURPLE)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Module Overview', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Essential code examples
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
ESSENTIAL NLTK CODE

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# 1. TOKENIZATION
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Apple stock rose 5%. Markets closed higher."

words = word_tokenize(text)
# ['Apple', 'stock', 'rose', '5', '%', '.', 'Markets', 'closed', 'higher', '.']

sentences = sent_tokenize(text)
# ['Apple stock rose 5%.', 'Markets closed higher.']


# 2. STOPWORD REMOVAL
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
filtered = [w for w in words if w.lower() not in stop_words]
# ['Apple', 'stock', 'rose', '5', '%', '.', 'Markets', 'closed', 'higher', '.']


# 3. STEMMING
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]


# 4. LEMMATIZATION
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(w) for w in filtered]


# 5. POS TAGGING
from nltk import pos_tag

tagged = pos_tag(words)
# [('Apple', 'NNP'), ('stock', 'NN'), ('rose', 'VBD'), ...]
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Essential Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete preprocessing pipeline
ax4 = axes[1, 1]
ax4.axis('off')

pipeline = '''
COMPLETE NLTK PREPROCESSING PIPELINE

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class NLTKPreprocessor:
    def __init__(self):
        # Download required data
        for resource in ['punkt', 'stopwords', 'wordnet']:
            nltk.download(resource, quiet=True)

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        # 1. Lowercase
        text = text.lower()

        # 2. Remove special characters (keep letters/numbers)
        text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)

        # 3. Tokenize
        tokens = word_tokenize(text)

        # 4. Remove stopwords and short words
        tokens = [t for t in tokens
                  if t not in self.stop_words and len(t) > 2]

        # 5. Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens


# USAGE
preprocessor = NLTKPreprocessor()

text = "The company's quarterly earnings exceeded analyst expectations."
tokens = preprocessor.preprocess(text)
print(tokens)
# ['company', 'quarterly', 'earnings', 'exceeded', 'analyst', 'expectation']


# BATCH PROCESSING
def preprocess_documents(documents):
    preprocessor = NLTKPreprocessor()
    return [preprocessor.preprocess(doc) for doc in documents]

# Example with financial headlines
headlines = [
    "Apple stock rises on strong earnings",
    "Fed raises interest rates by 25 basis points",
    "Tech sector leads market rally"
]
processed = preprocess_documents(headlines)
'''

ax4.text(0.02, 0.98, pipeline, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Pipeline', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
