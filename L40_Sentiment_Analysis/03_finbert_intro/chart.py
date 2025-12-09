"""FinBERT Introduction - Finance-Specific Sentiment"""
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
fig.suptitle('FinBERT: Finance-Specific Sentiment Analysis', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is FinBERT
ax1 = axes[0, 0]
ax1.axis('off')

finbert_intro = '''
WHAT IS FINBERT?

DEFINITION:
-----------
BERT model fine-tuned on financial text
for sentiment analysis.

BERT = Bidirectional Encoder Representations
       from Transformers (Google, 2018)


WHY FINBERT?
------------
General models fail on financial text!

"The company reported strong headwinds"
  VADER: neutral (misses "headwinds")
  FinBERT: negative (understands context)

"Bull market rally continues"
  VADER: negative (sees "bull" as animal)
  FinBERT: positive (knows finance meaning)


TRAINING DATA:
--------------
- Financial PhraseBank (4,840 sentences)
- Reuters financial news
- Analyst reports
- SEC filings


MODEL VARIANTS:
---------------
1. ProsusAI/finbert (most popular)
2. yiyanghkust/finbert-tone
3. ahmedrachid/FinancialBERT


OUTPUT:
-------
3-class: Positive, Negative, Neutral
With confidence probabilities


PERFORMANCE:
------------
Task              | VADER | FinBERT
------------------|-------|--------
Financial News    | 65%   | 87%
Earnings Calls    | 58%   | 82%
SEC Filings       | 52%   | 78%
'''

ax1.text(0.02, 0.98, finbert_intro, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is FinBERT?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: FinBERT vs VADER comparison
ax2 = axes[0, 1]

# Example comparisons
texts = [
    'Bull market rally',
    'Headwinds persist',
    'Short interest high',
    'Raised guidance',
    'Restructuring announced',
    'Aggressive growth'
]

vader_scores = [-0.34, -0.13, -0.11, 0.42, 0.0, -0.38]
finbert_scores = [0.85, -0.72, -0.45, 0.91, -0.55, 0.68]

x = np.arange(len(texts))
width = 0.35

bars1 = ax2.bar(x - width/2, vader_scores, width, label='VADER', color=MLBLUE, edgecolor='black')
bars2 = ax2.bar(x + width/2, finbert_scores, width, label='FinBERT', color=MLGREEN, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(texts, fontsize=8, rotation=20, ha='right')
ax2.set_ylabel('Sentiment Score')
ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax2.set_title('VADER vs FinBERT Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis='y')
ax2.set_ylim(-1, 1)

# Add "correct" annotations
correct_sentiments = ['Pos', 'Neg', 'Neg', 'Pos', 'Neg', 'Pos']
for i, (v, f, c) in enumerate(zip(vader_scores, finbert_scores, correct_sentiments)):
    is_correct_vader = (v > 0 and c == 'Pos') or (v < 0 and c == 'Neg')
    is_correct_finbert = (f > 0 and c == 'Pos') or (f < 0 and c == 'Neg')
    if not is_correct_vader and is_correct_finbert:
        ax2.annotate('*', xy=(i, max(v, f) + 0.1), fontsize=14, ha='center', color=MLRED)

ax2.text(5, 0.95, '* FinBERT correct,\n   VADER wrong', fontsize=8, ha='right')

# Plot 3: FinBERT code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
FINBERT USAGE IN PYTHON

# Installation
pip install transformers torch


# Basic Usage
from transformers import pipeline

# Load FinBERT pipeline
finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


# Analyze single text
text = "The company reported strong earnings growth"
result = finbert(text)
print(result)
# [{'label': 'positive', 'score': 0.9234}]


# Analyze multiple texts
texts = [
    "Revenue exceeded expectations",
    "Facing significant headwinds",
    "Quarterly results were mixed"
]

results = finbert(texts)
for text, result in zip(texts, results):
    print(f"{text[:35]:35} | {result['label']:8} ({result['score']:.2f})")

# Output:
# Revenue exceeded expectations        | positive (0.94)
# Facing significant headwinds         | negative (0.88)
# Quarterly results were mixed         | neutral  (0.67)


# Add to DataFrame
import pandas as pd

def get_finbert_sentiment(text):
    result = finbert(text)[0]
    return result['label'], result['score']

df['sentiment'], df['confidence'] = zip(*df['text'].apply(get_finbert_sentiment))


# BATCH PROCESSING (faster)
results = finbert(df['text'].tolist(), batch_size=32)
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('FinBERT Code Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: When to use what
ax4 = axes[1, 1]
ax4.axis('off')

comparison = '''
CHOOSING THE RIGHT TOOL

VADER - USE WHEN:
-----------------
+ Quick prototyping needed
+ Social media / tweets
+ Limited compute resources
+ No GPU available
+ General sentiment (non-finance)
+ Batch processing millions of texts

Speed: ~10,000 texts/second
Cost: Free, lightweight


FINBERT - USE WHEN:
-------------------
+ Financial text analysis
+ Earnings calls, SEC filings
+ News sentiment for trading
+ Accuracy is critical
+ GPU available

Speed: ~100 texts/second (GPU)
       ~10 texts/second (CPU)
Cost: Requires more compute


LOUGHRAN-MCDONALD - USE WHEN:
-----------------------------
+ Analyzing SEC filings
+ Need word-level sentiment
+ Regulatory / legal text
+ Explainability required

Speed: ~5,000 texts/second
Cost: Free, dictionary-based


HYBRID APPROACH:
----------------
1. Quick filter with VADER
2. Detailed analysis with FinBERT
3. Interpret with L-M dictionary

Example:
  if abs(vader_score) > 0.5:
      # Strong signal - trust VADER
  else:
      # Ambiguous - use FinBERT
'''

ax4.text(0.02, 0.98, comparison, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Choosing the Right Tool', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
