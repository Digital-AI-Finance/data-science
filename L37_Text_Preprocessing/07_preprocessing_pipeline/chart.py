"""Complete Preprocessing Pipeline"""
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
fig.suptitle('Complete Text Preprocessing Pipeline', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Pipeline overview
ax1 = axes[0, 0]
ax1.axis('off')

# Draw pipeline flowchart
steps = [
    ('Raw Text', MLBLUE),
    ('Clean HTML/URLs', MLORANGE),
    ('Lowercase', MLORANGE),
    ('Tokenize', MLORANGE),
    ('Remove Stopwords', MLORANGE),
    ('Stem/Lemmatize', MLORANGE),
    ('Clean Tokens', MLGREEN)
]

y_positions = np.linspace(0.9, 0.1, len(steps))

for i, (step, color) in enumerate(steps):
    y = y_positions[i]
    ax1.text(0.5, y, step, fontsize=11, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

    # Draw arrow to next step
    if i < len(steps) - 1:
        ax1.annotate('', xy=(0.5, y_positions[i+1] + 0.04),
                    xytext=(0.5, y - 0.04),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

ax1.text(0.5, 0.98, 'Preprocessing Pipeline', fontsize=12, ha='center',
         fontweight='bold', color=MLPURPLE)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('Pipeline Steps', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Pipeline configuration
ax2 = axes[0, 1]
ax2.axis('off')

config = '''
PIPELINE CONFIGURATION OPTIONS

STEP               | OPTIONS              | DEFAULT
-------------------|---------------------|----------
Lowercase          | True/False          | True
Remove HTML        | True/False          | True
Remove URLs        | True/False          | True
Remove emails      | True/False          | True
Remove numbers     | True/False          | False
Remove punctuation | True/False          | True
Tokenizer          | nltk/spacy/split    | nltk
Stopwords          | nltk/custom/none    | nltk
Normalization      | stem/lemma/none     | lemma
Min word length    | 1-5                 | 2


TASK-SPECIFIC RECOMMENDATIONS:
------------------------------

Sentiment Analysis:
- Keep: negations (not, no, never)
- Keep: intensifiers (very, really)
- Don't lowercase emojis/emoticons
- Use lemmatization

Topic Modeling:
- Remove all stopwords
- Lemmatization
- Remove rare words (< 5 occurrences)
- Remove common words (> 50% of docs)

Document Classification:
- Standard pipeline
- Lemmatization
- Keep numbers if relevant

Named Entity Recognition:
- Don't lowercase (case matters!)
- Don't remove punctuation
- Light preprocessing only
'''

ax2.text(0.02, 0.98, config, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Configuration Options', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Complete Python implementation
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
COMPLETE PREPROCESSING CLASS

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Optional


class TextPreprocessor:
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_word_length: int = 2,
        custom_stopwords: Optional[List[str]] = None
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length

        # Setup NLTK resources
        for resource in ['punkt', 'stopwords', 'wordnet']:
            nltk.download(resource, quiet=True)

        # Stopwords
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> List[str]:
        # 1. Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\\S+', '', text)

        # 2. Remove HTML
        text = re.sub(r'<[^>]+>', '', text)

        # 3. Lowercase
        if self.lowercase:
            text = text.lower()

        # 4. Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)

        # 5. Tokenize
        tokens = word_tokenize(text)

        # 6. Process tokens
        processed = []
        for token in tokens:
            # Skip short words
            if len(token) < self.min_word_length:
                continue
            # Skip numbers
            if self.remove_numbers and token.isdigit():
                continue
            # Skip stopwords
            if self.remove_stopwords and token in self.stop_words:
                continue
            # Lemmatize
            if self.lemmatize:
                token = self.lemmatizer.lemmatize(token)
            processed.append(token)

        return processed
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=6.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Usage examples
ax4 = axes[1, 1]
ax4.axis('off')

usage = '''
USAGE EXAMPLES

# Basic usage
preprocessor = TextPreprocessor()
text = "Apple's Q3 earnings exceeded analyst expectations!"
tokens = preprocessor.preprocess(text)
print(tokens)
# ['apple', 'q3', 'earnings', 'exceeded', 'analyst', 'expectation']


# Custom configuration
preprocessor = TextPreprocessor(
    lowercase=True,
    remove_numbers=False,
    remove_stopwords=True,
    lemmatize=True,
    min_word_length=3,
    custom_stopwords=['said', 'according', 'reported']
)


# Batch processing
def preprocess_corpus(documents: List[str]) -> List[List[str]]:
    preprocessor = TextPreprocessor()
    return [preprocessor.preprocess(doc) for doc in documents]

# Process financial headlines
headlines = [
    "Apple stock rises on strong earnings report",
    "Fed announces interest rate hike of 25 basis points",
    "Tech sector leads market rally amid optimism"
]

processed = preprocess_corpus(headlines)
for orig, proc in zip(headlines, processed):
    print(f"Original: {orig}")
    print(f"Processed: {proc}")
    print()


# Output:
# Original: Apple stock rises on strong earnings report
# Processed: ['apple', 'stock', 'rise', 'strong', 'earnings', 'report']

# Original: Fed announces interest rate hike of 25 basis points
# Processed: ['fed', 'announces', 'interest', 'rate', 'hike', 'basis', 'point']

# Original: Tech sector leads market rally amid optimism
# Processed: ['tech', 'sector', 'lead', 'market', 'rally', 'amid', 'optimism']


NEXT STEPS:
-----------
After preprocessing, convert to:
- Bag of Words (CountVectorizer)
- TF-IDF (TfidfVectorizer)
- Word embeddings (Word2Vec)
'''

ax4.text(0.02, 0.98, usage, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Usage Examples', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
