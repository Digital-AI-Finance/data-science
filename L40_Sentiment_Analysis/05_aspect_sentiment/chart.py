"""Aspect-Based Sentiment Analysis"""
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
fig.suptitle('Aspect-Based Sentiment Analysis (ABSA)', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is ABSA
ax1 = axes[0, 0]
ax1.axis('off')

absa_intro = '''
ASPECT-BASED SENTIMENT ANALYSIS

PROBLEM:
--------
Document-level sentiment misses nuance.

Example earnings call:
"Revenue growth was excellent, but margins
 disappointed due to rising costs."

Document sentiment: Neutral/Mixed
But this hides important information!


ABSA SOLUTION:
--------------
Extract sentiment for EACH ASPECT mentioned.

Aspects:
- Revenue growth -> Positive
- Margins -> Negative
- Costs -> Negative


COMPONENTS:
-----------
1. Aspect Extraction: Find topics mentioned
2. Sentiment Classification: Per aspect


FINANCIAL ASPECTS:
------------------
- Revenue / Sales
- Profit / Margins
- Costs / Expenses
- Guidance / Outlook
- Competition
- Demand / Market
- Management
- Products / Services


WHY ABSA MATTERS:
-----------------
Investors care about SPECIFIC aspects.
- Good revenue + bad margins = sell signal
- Bad revenue + improved margins = mixed signal

ABSA provides granular insights!
'''

ax1.text(0.02, 0.98, absa_intro, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is ABSA?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual example
ax2 = axes[0, 1]
ax2.axis('off')

# Draw ABSA example
text = '"Revenue exceeded expectations but\n costs remain elevated"'
ax2.text(0.5, 0.9, text, fontsize=11, ha='center', fontweight='bold',
         bbox=dict(facecolor=MLLAVENDER, alpha=0.6))

# Draw arrows to aspects
aspects = [
    ('Revenue', 0.25, 0.6, MLGREEN, 'Positive\n(+0.85)'),
    ('expectations', 0.5, 0.6, MLBLUE, 'Neutral'),
    ('costs', 0.75, 0.6, MLRED, 'Negative\n(-0.72)')
]

for aspect, x, y, color, sentiment in aspects:
    ax2.annotate('', xy=(x, 0.75), xytext=(x, y + 0.1),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
    ax2.text(x, y, aspect, fontsize=10, ha='center', fontweight='bold',
             bbox=dict(facecolor=color, alpha=0.4))
    ax2.text(x, y - 0.15, sentiment, fontsize=9, ha='center')

# Final output
ax2.text(0.5, 0.25, 'ASPECT SUMMARY:', fontsize=10, ha='center', fontweight='bold')
ax2.text(0.5, 0.15, 'Revenue: +0.85 | Costs: -0.72', fontsize=10, ha='center',
         bbox=dict(facecolor=MLLAVENDER, alpha=0.8))
ax2.text(0.5, 0.05, 'Overall: Mixed (+0.07)', fontsize=10, ha='center')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('ABSA Visual Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Simple ABSA implementation
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
SIMPLE ASPECT-BASED SENTIMENT

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re


# Define aspects to track
FINANCIAL_ASPECTS = {
    'revenue': ['revenue', 'sales', 'top-line', 'topline'],
    'profit': ['profit', 'earnings', 'income', 'margin'],
    'costs': ['cost', 'expense', 'spending'],
    'guidance': ['guidance', 'outlook', 'forecast', 'expect'],
    'growth': ['growth', 'increase', 'improve', 'gain']
}


def extract_aspect_sentences(text, aspects):
    \"\"\"Extract sentences mentioning each aspect.\"\"\"
    sentences = text.split('.')
    aspect_sentences = {aspect: [] for aspect in aspects}

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for aspect, keywords in aspects.items():
            if any(kw in sentence_lower for kw in keywords):
                aspect_sentences[aspect].append(sentence)

    return aspect_sentences


def aspect_sentiment(text, aspects=FINANCIAL_ASPECTS):
    \"\"\"Get sentiment for each aspect.\"\"\"
    sia = SentimentIntensityAnalyzer()
    aspect_sents = extract_aspect_sentences(text, aspects)

    results = {}
    for aspect, sentences in aspect_sents.items():
        if sentences:
            scores = [sia.polarity_scores(s)['compound'] for s in sentences]
            results[aspect] = {
                'sentiment': np.mean(scores),
                'count': len(sentences)
            }

    return results


# Example usage
text = \"\"\"
Revenue grew 15% year over year, exceeding analyst estimates.
However, operating margins declined due to rising costs.
Management provided cautious guidance for next quarter.
\"\"\"

results = aspect_sentiment(text)
# {'revenue': {'sentiment': 0.72, 'count': 1},
#  'profit': {'sentiment': -0.45, 'count': 1},
#  'costs': {'sentiment': -0.45, 'count': 1},
#  'guidance': {'sentiment': -0.23, 'count': 1}}
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Simple ABSA Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Aspect sentiment dashboard
ax4 = axes[1, 1]

# Simulated aspect sentiments over time
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
aspects_data = {
    'Revenue': [0.6, 0.7, 0.5, 0.8],
    'Margins': [0.3, -0.1, -0.3, -0.4],
    'Guidance': [0.4, 0.2, -0.1, 0.3],
    'Growth': [0.5, 0.6, 0.4, 0.7]
}

x = np.arange(len(quarters))
width = 0.2

colors = [MLBLUE, MLGREEN, MLORANGE, MLPURPLE]
for i, (aspect, values) in enumerate(aspects_data.items()):
    ax4.bar(x + i * width, values, width, label=aspect, color=colors[i], edgecolor='black')

ax4.set_xticks(x + 1.5 * width)
ax4.set_xticklabels(quarters)
ax4.set_ylabel('Sentiment Score')
ax4.set_xlabel('Quarter')
ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax4.set_title('Aspect Sentiment Over Time', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(alpha=0.3, axis='y')
ax4.set_ylim(-0.6, 1)

# Annotations
ax4.annotate('Margin\ndecline', xy=(2.4, -0.3), xytext=(3.2, -0.5),
            fontsize=8, arrowprops=dict(arrowstyle='->', color=MLRED))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
