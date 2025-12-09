"""Regular Expressions for Text Cleaning"""
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
fig.suptitle('Regular Expressions for Text Cleaning', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Regex basics
ax1 = axes[0, 0]
ax1.axis('off')

basics = '''
REGULAR EXPRESSIONS (REGEX) BASICS

WHAT IS REGEX?
--------------
A powerful pattern matching language for text.
Used to find, extract, or replace text patterns.


COMMON PATTERNS:
----------------
Pattern    | Meaning              | Example Match
-----------|---------------------|---------------
\\d         | Any digit           | 5, 0, 9
\\w         | Word char (a-z,0-9) | a, Z, 3
\\s         | Whitespace          | space, tab
.          | Any character       | a, 1, @
[abc]      | a, b, or c          | a
[^abc]     | NOT a, b, or c      | d, 1
a|b        | a OR b              | a, b


QUANTIFIERS:
------------
*          | 0 or more           | ab* -> a, ab, abb
+          | 1 or more           | ab+ -> ab, abb
?          | 0 or 1              | ab? -> a, ab
{n}        | Exactly n           | a{3} -> aaa
{n,m}      | n to m times        | a{2,4} -> aa, aaa


ANCHORS:
--------
^          | Start of string
$          | End of string
\\b         | Word boundary


PYTHON MODULE:
--------------
import re
re.search(pattern, text)  # Find first match
re.findall(pattern, text) # Find all matches
re.sub(pattern, repl, text) # Replace
'''

ax1.text(0.02, 0.98, basics, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Regex Basics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Common cleaning patterns
ax2 = axes[0, 1]
ax2.axis('off')

patterns = '''
COMMON TEXT CLEANING PATTERNS

import re


# 1. REMOVE URLs
text = "Check https://example.com for details"
clean = re.sub(r'https?://\\S+', '', text)
# "Check  for details"


# 2. REMOVE HTML TAGS
text = "<p>Stock <b>rose</b> 5%</p>"
clean = re.sub(r'<[^>]+>', '', text)
# "Stock rose 5%"


# 3. REMOVE EMAIL ADDRESSES
text = "Contact us at info@company.com"
clean = re.sub(r'\\S+@\\S+', '', text)
# "Contact us at "


# 4. REMOVE NUMBERS
text = "Price is 150.50 dollars"
clean = re.sub(r'\\d+\\.?\\d*', '', text)
# "Price is  dollars"


# 5. KEEP ONLY LETTERS
text = "Stock123 rose!!! @market"
clean = re.sub(r'[^a-zA-Z\\s]', '', text)
# "Stock rose market"


# 6. REMOVE EXTRA WHITESPACE
text = "Too    many   spaces"
clean = re.sub(r'\\s+', ' ', text).strip()
# "Too many spaces"


# 7. REMOVE PUNCTUATION
text = "Hello! How are you?"
clean = re.sub(r'[^\\w\\s]', '', text)
# "Hello How are you"


# 8. LOWERCASE AND CLEAN
def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text
'''

ax2.text(0.02, 0.98, patterns, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Common Cleaning Patterns', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Financial text patterns
ax3 = axes[1, 0]
ax3.axis('off')

finance = '''
FINANCE-SPECIFIC REGEX PATTERNS

import re


# 1. EXTRACT STOCK TICKERS ($AAPL, MSFT)
text = "Buy $AAPL and MSFT today"
tickers = re.findall(r'\\$?[A-Z]{1,5}\\b', text)
# ['AAPL', 'MSFT']


# 2. EXTRACT PRICES ($150.50, $1,234.00)
text = "Stock at $150.50, target $175.00"
prices = re.findall(r'\\$[\\d,]+\\.?\\d*', text)
# ['$150.50', '$175.00']


# 3. EXTRACT PERCENTAGES (+5.2%, -3.1%)
text = "Up +5.2% today, down -3.1% this week"
pcts = re.findall(r'[+-]?\\d+\\.?\\d*%', text)
# ['+5.2%', '-3.1%']


# 4. EXTRACT DATES
text = "Report on 12/31/2023 or 2023-12-31"
dates = re.findall(r'\\d{1,2}/\\d{1,2}/\\d{4}|\\d{4}-\\d{2}-\\d{2}', text)
# ['12/31/2023', '2023-12-31']


# 5. PRESERVE FINANCIAL TERMS
def clean_financial_text(text):
    # Placeholder for tickers and prices
    tickers = re.findall(r'\\$[A-Z]{1,5}\\b', text)
    prices = re.findall(r'\\$[\\d,]+\\.?\\d*', text)

    # Clean text
    clean = re.sub(r'[^a-zA-Z0-9\\s$%+-.]', '', text)
    clean = re.sub(r'\\s+', ' ', clean).strip()

    return clean


# 6. EXTRACT EARNINGS INFO
text = "Q3 EPS: $2.50, Revenue: $89.5B"
eps = re.search(r'EPS:\\s*\\$([\\d.]+)', text)
if eps:
    print(f"EPS: {eps.group(1)}")  # EPS: 2.50


# 7. CLEAN TWEET/SOCIAL MEDIA
def clean_social(text):
    text = re.sub(r'@\\w+', '', text)  # Remove mentions
    text = re.sub(r'#\\w+', '', text)   # Remove hashtags
    text = re.sub(r'RT\\s+', '', text)  # Remove RT
    text = re.sub(r'https?://\\S+', '', text)  # Remove URLs
    return text.strip()
'''

ax3.text(0.02, 0.98, finance, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Finance-Specific Patterns', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete cleaning function
ax4 = axes[1, 1]
ax4.axis('off')

complete = '''
COMPLETE TEXT CLEANING FUNCTION

import re
from typing import List, Dict


def clean_financial_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = False,
    keep_tickers: bool = True,
    keep_prices: bool = True,
    lowercase: bool = True
) -> str:
    """
    Clean financial text with configurable options.
    """
    # Save tickers and prices if needed
    saved = {}
    if keep_tickers:
        saved['tickers'] = re.findall(r'\\$[A-Z]{1,5}\\b', text)
    if keep_prices:
        saved['prices'] = re.findall(r'\\$[\\d,]+\\.?\\d*', text)

    # Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\\S+', '', text)

    # Remove emails
    if remove_emails:
        text = re.sub(r'\\S+@\\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove numbers (but keep prices/percentages)
    if remove_numbers:
        text = re.sub(r'(?<!\\$)\\b\\d+\\b(?!%)', '', text)

    # Remove special characters (keep $, %, .)
    text = re.sub(r'[^a-zA-Z0-9\\s$%.+-]', ' ', text)

    # Lowercase
    if lowercase:
        text = text.lower()

    # Clean whitespace
    text = re.sub(r'\\s+', ' ', text).strip()

    return text


# USAGE EXAMPLE
raw = "Check $AAPL at https://finance.yahoo.com! Price: $175.50 (+2.3%)"

clean = clean_financial_text(raw)
print(clean)
# "check $aapl price $175.50 +2.3%"

clean = clean_financial_text(raw, lowercase=False, keep_tickers=True)
print(clean)
# "Check $AAPL Price $175.50 +2.3%"
'''

ax4.text(0.02, 0.98, complete, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Cleaning Function', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
