"""TfidfVectorizer - sklearn Implementation"""
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
fig.suptitle('TfidfVectorizer: sklearn TF-IDF Implementation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic usage
ax1 = axes[0, 0]
ax1.axis('off')

basic = '''
TFIDFVECTORIZER BASICS

from sklearn.feature_extraction.text import TfidfVectorizer


# Basic usage
documents = [
    "stock price rose on earnings",
    "stock price fell after report",
    "market rallied on strong earnings",
    "investors concerned about earnings"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)


# View results
print(X.shape)  # (4, vocab_size)
print(X.toarray())  # TF-IDF matrix


# Get feature names
print(vectorizer.get_feature_names_out())


# IDF values for each word
for word, idx in vectorizer.vocabulary_.items():
    print(f"{word}: IDF = {vectorizer.idf_[idx]:.3f}")


# Transform new documents
new_doc = ["strong earnings report"]
X_new = vectorizer.transform(new_doc)


KEY DIFFERENCE FROM COUNTVECTORIZER:
------------------------------------
CountVectorizer: Raw word counts [1, 2, 0, 1, ...]
TfidfVectorizer: Weighted scores [0.2, 0.5, 0, 0.3, ...]

Values are normalized (L2 norm) by default!
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

from sklearn.feature_extraction.text import TfidfVectorizer


# FULL CONFIGURATION
vectorizer = TfidfVectorizer(
    # Vocabulary control
    max_features=5000,     # Limit vocabulary size
    min_df=2,              # Min document frequency
    max_df=0.95,           # Max document frequency (%)

    # Preprocessing
    lowercase=True,        # Convert to lowercase
    stop_words='english',  # Remove stopwords
    strip_accents='unicode',

    # TF-IDF specific
    norm='l2',             # Normalize vectors (l2, l1, None)
    use_idf=True,          # Use IDF weighting
    smooth_idf=True,       # Add 1 to prevent division by zero
    sublinear_tf=False,    # Use log(1 + tf) for TF

    # N-grams
    ngram_range=(1, 2),    # Unigrams and bigrams
)


SUBLINEAR TF:
-------------
sublinear_tf=True:
  Uses 1 + log(tf) instead of raw tf
  Reduces impact of very frequent terms
  Often improves results!


NORM OPTIONS:
-------------
'l2' (default): Each document vector has length 1
'l1': Vector elements sum to 1
None: No normalization

L2 is standard for most ML applications.


RECOMMENDED SETTINGS:
---------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    stop_words='english',
    ngram_range=(1, 2),
    sublinear_tf=True
)
'''

ax2.text(0.02, 0.98, params, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Important Parameters', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: CountVectorizer vs TfidfVectorizer comparison
ax3 = axes[1, 0]

# Sample words and their values
words = ['the', 'earnings', 'beat', 'algorithm', 'Q3']
count_values = [0.35, 0.25, 0.15, 0.05, 0.05]  # Normalized counts
tfidf_values = [0.05, 0.30, 0.35, 0.45, 0.40]  # TF-IDF scores

x = np.arange(len(words))
width = 0.35

bars1 = ax3.bar(x - width/2, count_values, width, label='CountVectorizer', color=MLBLUE, edgecolor='black')
bars2 = ax3.bar(x + width/2, tfidf_values, width, label='TfidfVectorizer', color=MLGREEN, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(words)
ax3.set_ylabel('Feature Value')
ax3.set_title('CountVectorizer vs TfidfVectorizer', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')

# Add annotations
ax3.annotate('Common words\ndownweighted', xy=(0, 0.15), xytext=(1, 0.55),
            arrowprops=dict(arrowstyle='->', color=MLRED), fontsize=8)
ax3.annotate('Rare words\nupweighted', xy=(3, 0.47), xytext=(2, 0.55),
            arrowprops=dict(arrowstyle='->', color=MLGREEN), fontsize=8)

# Plot 4: Complete example
ax4 = axes[1, 1]
ax4.axis('off')

example = '''
COMPLETE FINANCIAL CLASSIFICATION EXAMPLE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Financial headlines
headlines = [
    "Apple earnings beat expectations, stock soars",
    "Tech stocks tumble on interest rate fears",
    "Strong jobs report lifts market sentiment",
    "Oil prices crash amid supply concerns",
    "Investors optimistic about Q4 outlook",
    "Banks warn of recession risks ahead",
    "Retail sales surge during holiday season",
    "Market volatility spikes on uncertainty"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=bullish, 0=bearish


# Create TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(headlines)
print(f"Feature matrix shape: {X.shape}")


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)


# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)


# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# Top features per class
feature_names = vectorizer.get_feature_names_out()
for i, class_name in enumerate(['Bearish', 'Bullish']):
    top_idx = clf.coef_[0].argsort()[-(5-i*10):][::-1] if i else clf.coef_[0].argsort()[:5]
    top_features = [feature_names[j] for j in top_idx]
    print(f"{class_name}: {top_features}")
'''

ax4.text(0.02, 0.98, example, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
