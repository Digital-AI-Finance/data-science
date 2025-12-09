"""Finance Application - Domain-Specific Embeddings"""
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
fig.suptitle('Finance Application: Domain-Specific Embeddings', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why finance embeddings
ax1 = axes[0, 0]
ax1.axis('off')

why = '''
WHY FINANCE-SPECIFIC EMBEDDINGS?

PROBLEM WITH GENERAL EMBEDDINGS:
--------------------------------
General (Google News):
  "Apple" is similar to: fruit, banana, orange
  "Bull" is similar to: cow, animal, farm

Financial:
  "Apple" should be similar to: Microsoft, Google, tech
  "Bull" should be similar to: bullish, rally, uptrend


FINANCIAL VOCABULARY:
---------------------
Terms not in general corpora:
- P/E ratio, EBITDA, VaR
- Basis points, yield curve
- ITM, OTM (options)
- Many ticker symbols


DOMAIN RELATIONSHIPS:
---------------------
General: stock ~ inventory ~ supply
Finance: stock ~ equity ~ share

General: bear ~ animal ~ forest
Finance: bear ~ bearish ~ selloff


DATA SOURCES FOR TRAINING:
--------------------------
- SEC filings (10-K, 10-Q)
- Earnings call transcripts
- Financial news (Reuters, Bloomberg)
- Analyst reports
- Trading forums


RESULT:
-------
Finance embeddings capture:
- Financial term relationships
- Market concepts
- Company similarities
'''

ax1.text(0.02, 0.98, why, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Why Finance Embeddings?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Similarity comparison
ax2 = axes[0, 1]

# Simulated similarities for "stock"
words = ['equity', 'share', 'bond', 'market', 'inventory', 'supply']
general_sim = [0.45, 0.40, 0.35, 0.55, 0.65, 0.60]
finance_sim = [0.92, 0.88, 0.72, 0.78, 0.25, 0.20]

x = np.arange(len(words))
width = 0.35

bars1 = ax2.bar(x - width/2, general_sim, width, label='General Embeddings', color=MLBLUE, edgecolor='black')
bars2 = ax2.bar(x + width/2, finance_sim, width, label='Finance Embeddings', color=MLGREEN, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(words, fontsize=8, rotation=15)
ax2.set_ylabel('Similarity to "stock"')
ax2.set_title('General vs Finance Embeddings', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis='y')

ax2.text(4.5, 0.55, 'Finance model\ncorrectly ranks\nfinancial terms', fontsize=8,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Training code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
TRAINING FINANCE EMBEDDINGS

import os
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


# 1. COLLECT DATA
# Assume we have financial documents
documents = load_financial_corpus()
# - SEC filings
# - News articles
# - Earnings transcripts


# 2. PREPROCESS
def preprocess(text):
    text = text.lower()
    # Keep financial terms
    tokens = word_tokenize(text)
    # Remove very short tokens
    return [t for t in tokens if len(t) > 2]

sentences = [preprocess(doc) for doc in documents]
print(f"Total sentences: {len(sentences)}")


# 3. TRAIN MODEL
model = Word2Vec(
    sentences=sentences,
    vector_size=200,        # Finance: 200-300 works well
    window=10,              # Larger window for documents
    min_count=5,            # Remove rare terms
    workers=4,
    epochs=15,
    sg=1                    # Skip-gram for rare terms
)


# 4. EVALUATE
# Check financial relationships
print(model.wv.most_similar('stock', topn=5))
# Should see: equity, share, market, etc.

print(model.wv.most_similar('bullish', topn=5))
# Should see: rally, uptrend, optimistic, etc.

# Test analogy
result = model.wv.most_similar(
    positive=['profit', 'negative'],
    negative=['positive'],
    topn=1
)
print(f"profit - positive + negative = {result}")


# 5. SAVE
model.save('finance_word2vec.model')


# 6. USE FOR DOWNSTREAM TASKS
# Document classification, sentiment, etc.
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Training Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Applications
ax4 = axes[1, 1]
ax4.axis('off')

applications = '''
APPLICATIONS OF FINANCE EMBEDDINGS

1. NEWS CLASSIFICATION:
-----------------------
Classify financial news by topic/sentiment.
Finance embeddings improve accuracy 5-10%!


2. SIMILAR COMPANY SEARCH:
--------------------------
# Find companies similar to Apple
doc_vec = get_company_description_vector("Apple")
similar = find_similar_companies(doc_vec, topn=5)
# Microsoft, Google, Amazon, Meta, ...


3. EARNINGS CALL ANALYSIS:
--------------------------
Analyze tone and topics in earnings calls.
Compare language patterns across quarters.


4. RISK DOCUMENT ANALYSIS:
--------------------------
Find similar risk factors across 10-K filings.
Track how risk language evolves.


5. TRADING SIGNAL FEATURES:
---------------------------
Use embedding-based features for models:
- News sentiment vectors
- Document similarity to positive/negative templates
- Topic vectors


PERFORMANCE IMPROVEMENTS:
-------------------------
Task                  | General | Finance | Gain
---------------------|---------|---------|------
News Classification   | 78%     | 85%     | +7%
Sentiment Analysis    | 72%     | 81%     | +9%
Entity Recognition    | 80%     | 86%     | +6%


AVAILABLE FINANCE EMBEDDINGS:
-----------------------------
- FinBERT (transformers-based)
- Finance-specific Word2Vec (train your own)
- Loughran-McDonald sentiment lexicon
'''

ax4.text(0.02, 0.98, applications, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Applications', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
