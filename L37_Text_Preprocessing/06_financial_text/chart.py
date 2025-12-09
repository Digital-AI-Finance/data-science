"""Financial Text Characteristics"""
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
fig.suptitle('Financial Text: Unique Characteristics', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Financial text characteristics
ax1 = axes[0, 0]
ax1.axis('off')

characteristics = '''
FINANCIAL TEXT CHARACTERISTICS

UNIQUE ELEMENTS:
----------------
1. Stock tickers: AAPL, MSFT, $GOOG
2. Prices: $150.50, EUR 125.00
3. Percentages: +5.2%, -3.1%, 15%
4. Financial ratios: P/E, EPS, ROI
5. Dates/quarters: Q3 2023, FY2024
6. Large numbers: $2.5B, 1.2M shares
7. Abbreviations: CEO, IPO, M&A, SEC


TYPES OF FINANCIAL TEXT:
------------------------
- News headlines
- SEC filings (10-K, 10-Q)
- Earnings calls (transcripts)
- Analyst reports
- Social media ($AAPL trending)
- Trading signals


CHALLENGES:
-----------
1. Domain-specific vocabulary
2. Numbers carry meaning (can't just remove)
3. Sentiment is nuanced
   "beat expectations" = positive
   "missed estimates" = negative

4. Time sensitivity
5. Forward-looking statements


PREPROCESSING PRIORITIES:
-------------------------
- Keep tickers intact
- Preserve numbers/percentages
- Handle financial abbreviations
- Maintain temporal information
'''

ax1.text(0.02, 0.98, characteristics, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Financial Text Characteristics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Common financial terms frequency
ax2 = axes[0, 1]

terms = ['earnings', 'revenue', 'stock', 'market', 'growth',
         'profit', 'shares', 'analyst', 'quarter', 'price']
frequencies = [850, 720, 680, 650, 520, 480, 450, 420, 400, 380]

colors = [MLGREEN if i < 5 else MLBLUE for i in range(len(terms))]

bars = ax2.barh(range(len(terms)), frequencies, color=colors, edgecolor='black')
ax2.set_yticks(range(len(terms)))
ax2.set_yticklabels(terms)
ax2.set_xlabel('Frequency (per 100 articles)')
ax2.set_title('Common Financial Terms', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.invert_yaxis()
ax2.grid(alpha=0.3, axis='x')

ax2.text(600, 8, 'Top 10 most\nfrequent terms\nin financial news', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Financial NLP pipeline
ax3 = axes[1, 0]
ax3.axis('off')

pipeline = '''
FINANCIAL TEXT PREPROCESSING PIPELINE

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class FinancialTextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Keep financial terms
        self.stop_words -= {'up', 'down', 'above', 'below', 'over', 'under'}

        # Financial abbreviations to preserve
        self.preserve = {'ceo', 'cfo', 'ipo', 'eps', 'pe', 'roi', 'ebitda',
                        'sec', 'fed', 'gdp', 'etf', 'nyse', 'nasdaq'}

    def preprocess(self, text):
        # 1. Extract and save important elements
        tickers = re.findall(r'\\$?[A-Z]{2,5}\\b', text)
        prices = re.findall(r'\\$[\\d,]+\\.?\\d*', text)
        percentages = re.findall(r'[+-]?\\d+\\.?\\d*%', text)

        # 2. Basic cleaning
        text = text.lower()
        text = re.sub(r'https?://\\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)

        # 3. Tokenize
        tokens = word_tokenize(text)

        # 4. Filter
        tokens = [t for t in tokens
                  if (t not in self.stop_words or t in self.preserve)
                  and (t.isalpha() or t in self.preserve)]

        # 5. Add back special elements
        return {
            'tokens': tokens,
            'tickers': tickers,
            'prices': prices,
            'percentages': percentages
        }


# USAGE
preprocessor = FinancialTextPreprocessor()
text = "Apple ($AAPL) stock rose 5.2% to $175.50 after CEO announced..."
result = preprocessor.preprocess(text)

print(result)
# {'tokens': ['apple', 'stock', 'rose', 'ceo', 'announced'],
#  'tickers': ['AAPL'],
#  'prices': ['$175.50'],
#  'percentages': ['5.2%']}
'''

ax3.text(0.02, 0.98, pipeline, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Financial Preprocessing Pipeline', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Examples of financial text processing
ax4 = axes[1, 1]
ax4.axis('off')

examples = '''
FINANCIAL TEXT PROCESSING EXAMPLES

EXAMPLE 1 - News Headline:
--------------------------
Input:  "AAPL Surges 8% on iPhone 15 Launch; Analyst Upgrades to Buy"

Processing:
- Ticker: AAPL
- Percentage: 8%
- Sentiment words: surges, upgrades, buy

Output: ['aapl', 'surges', 'iphone', 'launch', 'analyst', 'upgrades', 'buy']


EXAMPLE 2 - Earnings Report:
----------------------------
Input:  "Q3 EPS of $2.85 beat estimates by $0.12; Revenue up 15% YoY"

Processing:
- Quarter: Q3
- EPS: $2.85
- Beat: positive sentiment
- Revenue growth: +15%

Output: ['eps', 'beat', 'estimates', 'revenue', 'yoy']


EXAMPLE 3 - SEC Filing:
-----------------------
Input:  "The Company recorded goodwill impairment of $1.2B related
         to the 2023 acquisition of XYZ Corp."

Processing:
- Amount: $1.2B
- Term: goodwill impairment (negative)
- Company: XYZ Corp

Output: ['company', 'recorded', 'goodwill', 'impairment',
         'related', 'acquisition', 'xyz', 'corp']


EXAMPLE 4 - Social Media:
-------------------------
Input:  "$TSLA to the moon! CEO Elon says Q4 deliveries 🚀"

Processing:
- Ticker: $TSLA
- Remove emoji
- Sentiment: positive (moon = bullish slang)

Output: ['tsla', 'moon', 'ceo', 'elon', 'q4', 'deliveries']


KEY TAKEAWAYS:
--------------
1. Financial text needs special handling
2. Numbers and symbols carry meaning
3. Domain vocabulary is essential
4. Sentiment is context-dependent
'''

ax4.text(0.02, 0.98, examples, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Processing Examples', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
