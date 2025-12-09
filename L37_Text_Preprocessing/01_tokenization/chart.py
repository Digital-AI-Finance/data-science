"""Tokenization - Breaking Text into Pieces"""
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
fig.suptitle('Tokenization: Breaking Text into Pieces', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is tokenization
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT IS TOKENIZATION?

DEFINITION:
-----------
Splitting text into smaller units called "tokens".
Tokens are the basic building blocks for NLP.


TYPES OF TOKENS:
----------------
1. Word tokens: "Apple" "stock" "rose" "5%"
2. Sentence tokens: ["Apple stock rose.", "Markets closed higher."]
3. Character tokens: "A" "p" "p" "l" "e"
4. Subword tokens: "un" "##believ" "##able" (for deep learning)


EXAMPLE:
--------
Input: "Apple's Q3 earnings beat expectations by 15%."

Word tokens:
["Apple", "'s", "Q3", "earnings", "beat", "expectations", "by", "15", "%", "."]


WHY TOKENIZE?
-------------
- Computers can't understand raw text
- Need discrete units for counting/analysis
- Foundation for all NLP tasks
- Enables vocabulary creation


CHALLENGES:
-----------
- Contractions: "don't" -> "do" + "n't" or "don't"?
- Punctuation: "$5.00" -> "$" "5" "." "00"?
- Multi-word: "New York" -> one token or two?
- Numbers: "15%" -> "15" + "%" or "15%"?
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is Tokenization?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual tokenization example
ax2 = axes[0, 1]
ax2.axis('off')

# Draw tokenization flow
text = "Stock prices fell 2.5% today"
tokens = ["Stock", "prices", "fell", "2.5", "%", "today"]

# Original text
ax2.text(0.5, 0.9, text, fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=MLBLUE, alpha=0.3))
ax2.text(0.5, 0.95, 'Input Text', fontsize=10, ha='center', color='gray')

# Arrow
ax2.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.8),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=3))
ax2.text(0.55, 0.72, 'Tokenize', fontsize=10, color=MLORANGE)

# Tokens
x_positions = np.linspace(0.1, 0.9, len(tokens))
for i, (x, token) in enumerate(zip(x_positions, tokens)):
    ax2.text(x, 0.5, token, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor=MLGREEN, alpha=0.5))

ax2.text(0.5, 0.35, f'{len(tokens)} tokens', fontsize=12, ha='center', fontweight='bold')

# Show different tokenization results
ax2.text(0.5, 0.2, 'Different tokenizers give different results!', fontsize=10, ha='center', style='italic')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Tokenization Visualization', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Python code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
TOKENIZATION IN PYTHON

# METHOD 1: Simple split (basic)
text = "Stock prices fell 2.5% today"
tokens = text.split()  # ['Stock', 'prices', 'fell', '2.5%', 'today']


# METHOD 2: NLTK word tokenizer (recommended)
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Apple's Q3 earnings beat expectations. Stock rose 5%."

# Word tokenization
word_tokens = word_tokenize(text)
# ['Apple', "'s", 'Q3', 'earnings', 'beat', 'expectations', '.',
#  'Stock', 'rose', '5', '%', '.']

# Sentence tokenization
sent_tokens = sent_tokenize(text)
# ['Apple's Q3 earnings beat expectations.', 'Stock rose 5%.']


# METHOD 3: Regular expressions (custom)
import re
text = "Price: $150.50 (+3.2%)"
tokens = re.findall(r'\\w+|[\\$%+.-]', text)
# ['Price', ':', '$', '150', '.', '50', '+', '3', '.', '2', '%']


# METHOD 4: spaCy (industrial strength)
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple's stock rose 5%.")
tokens = [token.text for token in doc]
# ['Apple', "'s", 'stock', 'rose', '5', '%', '.']


# CHOOSING A TOKENIZER:
# - Simple tasks: nltk word_tokenize
# - Production: spaCy
# - Deep learning: subword tokenizers (BPE, WordPiece)
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Finance-specific tokenization challenges
ax4 = axes[1, 1]
ax4.axis('off')

finance = '''
FINANCE-SPECIFIC TOKENIZATION

CHALLENGES:
-----------
1. Stock tickers: "AAPL" "$AAPL" "@AAPL"
2. Currency: "$150.50" "EUR/USD"
3. Percentages: "+3.5%" "-2.1%"
4. Numbers: "Q3" "FY2023" "10-K"
5. Abbreviations: "CEO" "IPO" "P/E"


FINANCIAL TEXT EXAMPLES:
------------------------
"AAPL closed at $175.50 (+2.3%)"

Default tokenizer:
['AAPL', 'closed', 'at', '$', '175', '.', '50', '+', '2', '.', '3', '%']

Better for finance:
['AAPL', 'closed', 'at', '$175.50', '+2.3%']


CUSTOM FINANCE TOKENIZER:
-------------------------
import re

def finance_tokenize(text):
    # Keep prices and percentages together
    pattern = r'''
        \\$[\\d,.]+        # Prices: $150.50
        |[+-]?\\d+\\.?\\d*% # Percentages: +3.5%
        |\\b[A-Z]{1,5}\\b   # Tickers: AAPL
        |\\w+              # Regular words
    '''
    return re.findall(pattern, text, re.VERBOSE)

text = "AAPL rose $2.50 (+1.5%)"
print(finance_tokenize(text))
# ['AAPL', 'rose', '$2.50', '+1.5%']


TIP:
----
Always inspect your tokens!
print(tokens[:20]) after tokenizing.
'''

ax4.text(0.02, 0.98, finance, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Finance-Specific Challenges', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
