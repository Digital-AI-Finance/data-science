"""Sentiment Analysis Concept - Introduction to Opinion Mining"""
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
fig.suptitle('Sentiment Analysis: Introduction to Opinion Mining', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is sentiment analysis
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT IS SENTIMENT ANALYSIS?

DEFINITION:
-----------
Automatically determining the emotional tone
or opinion expressed in text.

Also called: Opinion Mining, Emotion AI


SENTIMENT CATEGORIES:
---------------------
POSITIVE: "Excellent results!"
NEGATIVE: "Disappointing quarter"
NEUTRAL:  "Revenue was $5B"


GRANULARITY LEVELS:
-------------------
1. Document-level: Overall sentiment
2. Sentence-level: Per-sentence sentiment
3. Aspect-level: Sentiment toward specific topics


APPLICATIONS IN FINANCE:
------------------------
- Earnings call analysis
- News sentiment scoring
- Social media monitoring
- Analyst report analysis
- Customer feedback


OUTPUT FORMATS:
---------------
Binary:    Positive / Negative
3-class:   Positive / Neutral / Negative
Score:     -1.0 (very neg) to +1.0 (very pos)
Compound:  Aggregated score from multiple signals


WHY IT MATTERS:
---------------
Market sentiment drives prices.
Quantifying sentiment = trading signals.
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is Sentiment Analysis?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Sentiment spectrum visualization
ax2 = axes[0, 1]

# Create sentiment spectrum
examples = [
    ('Terrible losses', -0.9),
    ('Disappointing results', -0.5),
    ('Below expectations', -0.3),
    ('Revenue was $5B', 0.0),
    ('Solid performance', 0.4),
    ('Strong growth', 0.7),
    ('Record profits!', 0.95)
]

texts, scores = zip(*examples)
colors = [MLRED if s < -0.2 else MLGREEN if s > 0.2 else MLBLUE for s in scores]

y_pos = np.arange(len(texts))
bars = ax2.barh(y_pos, scores, color=colors, edgecolor='black')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(texts, fontsize=9)
ax2.set_xlabel('Sentiment Score')
ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax2.axvline(-0.2, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlim(-1, 1)
ax2.set_title('Sentiment Spectrum', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3, axis='x')

# Add region labels
ax2.text(-0.6, 7.5, 'NEGATIVE', fontsize=10, color=MLRED, ha='center', fontweight='bold')
ax2.text(0, 7.5, 'NEUTRAL', fontsize=10, color=MLBLUE, ha='center', fontweight='bold')
ax2.text(0.6, 7.5, 'POSITIVE', fontsize=10, color=MLGREEN, ha='center', fontweight='bold')

# Plot 3: Approaches comparison
ax3 = axes[1, 0]
ax3.axis('off')

approaches = '''
SENTIMENT ANALYSIS APPROACHES

1. LEXICON-BASED (Rule-based):
------------------------------
Use pre-built sentiment dictionaries.
Count positive/negative words.

Pros: Fast, interpretable, no training
Cons: Misses context, domain-specific

Examples: VADER, Loughran-McDonald


2. MACHINE LEARNING:
--------------------
Train classifier on labeled data.
Features: TF-IDF, n-grams, embeddings

Pros: Can learn domain patterns
Cons: Needs labeled data

Examples: Naive Bayes, SVM, Random Forest


3. DEEP LEARNING:
-----------------
Neural networks on text sequences.
Learns features automatically.

Pros: State-of-the-art accuracy
Cons: Needs lots of data, slower

Examples: LSTM, BERT, FinBERT


FINANCE-SPECIFIC TOOLS:
-----------------------
- Loughran-McDonald dictionary (finance-tuned)
- FinBERT (BERT fine-tuned on finance)
- Custom dictionaries per domain


RECOMMENDATION:
---------------
Start: VADER (general) or L-M (finance)
Better: Train classifier with domain data
Best: FinBERT for state-of-the-art
'''

ax3.text(0.02, 0.98, approaches, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Approaches Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Financial sentiment examples
ax4 = axes[1, 1]
ax4.axis('off')

finance_examples = '''
FINANCIAL SENTIMENT EXAMPLES

POSITIVE INDICATORS:
--------------------
"beat expectations" -> Strong positive
"raised guidance" -> Very positive
"strong demand" -> Positive
"exceeded forecasts" -> Positive
"dividend increase" -> Positive


NEGATIVE INDICATORS:
--------------------
"missed estimates" -> Strong negative
"lowered outlook" -> Very negative
"declining margins" -> Negative
"headwinds" -> Negative
"restructuring" -> Often negative


TRICKY CASES (Context matters!):
--------------------------------
"volatile" -> Negative (usually)
"aggressive" -> Context-dependent
"limited downside" -> Positive!
"risk" -> Often neutral in finance

"The stock fell 10%"
  -> Factual, but negative signal

"Despite challenges, profits rose"
  -> Mixed, overall positive


DOMAIN KNOWLEDGE REQUIRED:
--------------------------
General: "Bull" = animal
Finance: "Bull" = positive market

General: "Short" = length
Finance: "Short" = bearish bet

This is why FINANCE-SPECIFIC
sentiment tools matter!
'''

ax4.text(0.02, 0.98, finance_examples, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Financial Sentiment Examples', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
