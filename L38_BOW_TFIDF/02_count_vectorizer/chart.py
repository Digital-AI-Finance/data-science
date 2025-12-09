"""CountVectorizer - sklearn Implementation"""
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
fig.suptitle('CountVectorizer: sklearn Bag of Words', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic usage
ax1 = axes[0, 0]
ax1.axis('off')

basic = '''
COUNTVECTORIZER BASICS

from sklearn.feature_extraction.text import CountVectorizer


# Basic usage
documents = [
    "stock price rose today",
    "stock price fell sharply",
    "market rallied on earnings"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)


# View results
print(X.toarray())
# [[0 0 1 0 1 1 0 1]
#  [0 1 0 0 1 0 1 1]
#  [1 0 0 1 0 0 0 0]]

print(vectorizer.get_feature_names_out())
# ['earnings' 'fell' 'price' 'rallied' 'rose' 'sharply' 'stock' 'today']


# Vocabulary mapping
print(vectorizer.vocabulary_)
# {'stock': 6, 'price': 2, 'rose': 4, 'today': 7, ...}


# Transform new documents
new_doc = ["stock market rose"]
X_new = vectorizer.transform(new_doc)


KEY METHODS:
------------
fit(docs)           - Learn vocabulary
transform(docs)     - Convert to vectors
fit_transform(docs) - Both in one step
get_feature_names_out() - Get vocabulary list
'''

ax1.text(0.02, 0.98, basic, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Basic Usage', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Important parameters
ax2 = axes[0, 1]
ax2.axis('off')

params = '''
IMPORTANT PARAMETERS

from sklearn.feature_extraction.text import CountVectorizer


# 1. CONTROL VOCABULARY SIZE
vectorizer = CountVectorizer(
    max_features=1000,     # Keep top 1000 words
    min_df=5,              # Word must appear in >= 5 docs
    max_df=0.8,            # Word must appear in <= 80% docs
)


# 2. PREPROCESSING OPTIONS
vectorizer = CountVectorizer(
    lowercase=True,        # Convert to lowercase
    strip_accents='unicode', # Remove accents
    stop_words='english',  # Remove English stopwords
)


# 3. CUSTOM TOKENIZATION
def custom_tokenizer(text):
    # Your tokenization logic
    return text.lower().split()

vectorizer = CountVectorizer(
    tokenizer=custom_tokenizer,
    token_pattern=None     # Disable default pattern
)


# 4. CHARACTER N-GRAMS
vectorizer = CountVectorizer(
    analyzer='char',       # Character-level
    ngram_range=(2, 4)     # 2 to 4 character sequences
)


# 5. BINARY COUNTS (presence only)
vectorizer = CountVectorizer(
    binary=True            # 1 if word present, 0 otherwise
)


RECOMMENDED SETTINGS:
---------------------
General: max_features=5000, min_df=2, max_df=0.95
Small dataset: max_features=1000, min_df=1
Large dataset: max_features=10000, min_df=5
'''

ax2.text(0.02, 0.98, params, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Important Parameters', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Effect of parameters visualization
ax3 = axes[1, 0]

# Show effect of min_df/max_df
vocab_sizes = [50000, 35000, 20000, 10000, 5000]
labels = ['Raw', 'min_df=2', 'min_df=5', '+max_df=0.9', '+max_features=5000']

colors = [MLRED, MLORANGE, MLORANGE, MLGREEN, MLGREEN]
bars = ax3.bar(range(len(vocab_sizes)), vocab_sizes, color=colors, edgecolor='black')

ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
ax3.set_ylabel('Vocabulary Size')
ax3.set_title('Effect of Parameters on Vocabulary', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.grid(alpha=0.3, axis='y')

# Add value labels
for bar, size in zip(bars, vocab_sizes):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'{size:,}', ha='center', fontsize=9)

ax3.text(2, 40000, 'Smaller vocab =\nFaster training', fontsize=9, ha='center',
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 4: Complete example
ax4 = axes[1, 1]
ax4.axis('off')

example = '''
COMPLETE FINANCIAL TEXT EXAMPLE

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# Sample financial headlines
headlines = [
    "Apple stock rises on strong earnings",
    "Tech stocks fall amid market uncertainty",
    "Fed raises interest rates by 25 basis points",
    "Markets rally on positive economic data",
    "Oil prices surge after OPEC decision",
    "Earnings beat expectations, stock jumps",
    "Market drops on inflation concerns",
    "Growth stocks lead market recovery"
]

labels = [1, 0, 0, 1, 1, 1, 0, 1]  # 1=positive, 0=negative


# Create vectorizer
vectorizer = CountVectorizer(
    max_features=100,       # Limit vocabulary
    stop_words='english',   # Remove stopwords
    lowercase=True
)


# Transform text to vectors
X = vectorizer.fit_transform(headlines)

print(f"Matrix shape: {X.shape}")  # (8, ~30)
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


# View top words
feature_names = vectorizer.get_feature_names_out()
print(f"Features: {list(feature_names)[:10]}")


# Check sparsity
sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))
print(f"Sparsity: {sparsity:.1%}")


# Ready for ML!
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)

# Next: fit classifier on X_train
'''

ax4.text(0.02, 0.98, example, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
