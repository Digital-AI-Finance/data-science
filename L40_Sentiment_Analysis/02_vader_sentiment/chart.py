"""VADER Sentiment Analyzer"""
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
fig.suptitle('VADER Sentiment Analyzer', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is VADER
ax1 = axes[0, 0]
ax1.axis('off')

vader_intro = '''
VADER (Valence Aware Dictionary and sEntiment Reasoner)

WHAT IS VADER?
--------------
Rule-based sentiment analysis tool
Specifically tuned for social media text
Part of NLTK library

KEY FEATURES:
-------------
- Handles slang, emojis, punctuation
- Considers capitalization (GREAT vs great)
- Handles negation (not good)
- Degree modifiers (very, extremely)
- Conjunctions (but, however)


OUTPUT SCORES:
--------------
neg:      Negative sentiment (0 to 1)
neu:      Neutral sentiment (0 to 1)
pos:      Positive sentiment (0 to 1)
compound: Aggregated score (-1 to +1)

Note: neg + neu + pos = 1


COMPOUND SCORE INTERPRETATION:
------------------------------
compound >= 0.05  -> Positive
compound <= -0.05 -> Negative
-0.05 < compound < 0.05 -> Neutral


INSTALLATION:
-------------
pip install nltk
import nltk
nltk.download('vader_lexicon')


BEST FOR:
---------
- Social media analysis
- Quick sentiment scoring
- When no training data available
'''

ax1.text(0.02, 0.98, vader_intro, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is VADER?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: VADER code example
ax2 = axes[0, 1]
ax2.axis('off')

code = '''
VADER USAGE IN PYTHON

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')


# Initialize analyzer
sia = SentimentIntensityAnalyzer()


# Analyze single text
text = "The stock performance was absolutely amazing!"
scores = sia.polarity_scores(text)

print(scores)
# {'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.75}


# Analyze multiple texts
texts = [
    "Great earnings beat!",
    "Disappointing results",
    "Revenue was $5 billion"
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"{text[:30]:30} -> {scores['compound']:+.2f}")

# Output:
# Great earnings beat!           -> +0.66
# Disappointing results          -> -0.50
# Revenue was $5 billion         -> +0.00


# Create sentiment column in DataFrame
import pandas as pd

df['sentiment'] = df['headline'].apply(
    lambda x: sia.polarity_scores(x)['compound']
)

# Classify based on compound score
df['sentiment_class'] = df['sentiment'].apply(
    lambda x: 'positive' if x >= 0.05
              else 'negative' if x <= -0.05
              else 'neutral'
)
'''

ax2.text(0.02, 0.98, code, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('VADER Code Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: VADER special features
ax3 = axes[1, 0]

# Show how VADER handles different text features
features = [
    ('good', 0.44),
    ('GOOD', 0.54),
    ('good!!!', 0.65),
    ('GOOD!!!', 0.74),
    ('very good', 0.62),
    ('extremely good', 0.72),
    ('not good', -0.34),
    ('not very good', -0.47),
]

texts, scores = zip(*features)
colors = [MLGREEN if s > 0 else MLRED for s in scores]

y_pos = np.arange(len(texts))
bars = ax3.barh(y_pos, scores, color=colors, edgecolor='black')

ax3.set_yticks(y_pos)
ax3.set_yticklabels(texts, fontsize=10)
ax3.set_xlabel('Compound Score')
ax3.axvline(0, color='black', linestyle='-', linewidth=1)
ax3.set_xlim(-0.8, 0.9)
ax3.set_title('VADER Handles Text Features', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.grid(alpha=0.3, axis='x')

# Add annotations
ax3.annotate('Capitalization\nincreases\nintensity', xy=(0.54, 1), xytext=(0.75, 0.5),
            fontsize=8, arrowprops=dict(arrowstyle='->', color=MLORANGE))
ax3.annotate('Punctuation\nincreases\nintensity', xy=(0.65, 2), xytext=(0.75, 2),
            fontsize=8, arrowprops=dict(arrowstyle='->', color=MLORANGE))
ax3.annotate('Negation\nflips\nsentiment', xy=(-0.34, 6), xytext=(-0.7, 5),
            fontsize=8, arrowprops=dict(arrowstyle='->', color=MLORANGE))

# Plot 4: VADER limitations
ax4 = axes[1, 1]
ax4.axis('off')

limitations = '''
VADER LIMITATIONS IN FINANCE

DOMAIN MISMATCH:
----------------
VADER is trained on general/social media text.
Finance has different word meanings!

"Bull market" -> VADER sees "bull" (animal)
"Short position" -> VADER misses financial meaning
"Bearish outlook" -> Partial understanding


EXAMPLES OF VADER FAILURES:
---------------------------
Text                    | VADER  | Actual
------------------------|--------|--------
"Bull run continues"    | -0.34  | Positive
"Shorts are squeezed"   | -0.11  | Positive
"Aggressive growth"     | -0.38  | Positive
"Stock fell 5%"         | -0.26  | Negative*
"Risk-adjusted returns" |  0.00  | Neutral

* VADER catches "fell" but not the financial context


WHEN VADER WORKS:
-----------------
+ Clear positive/negative words
+ Social media sentiment
+ General news headlines
+ Customer feedback


WHEN TO USE ALTERNATIVES:
-------------------------
- Earnings calls (use FinBERT)
- SEC filings (use Loughran-McDonald)
- Technical analysis text
- Domain-specific terminology


RECOMMENDATION:
---------------
VADER is a great STARTING POINT.
For production finance systems,
consider domain-specific tools.
'''

ax4.text(0.02, 0.98, limitations, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('VADER Limitations in Finance', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
