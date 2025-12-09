"""Finance Application - News Article Cleaning"""
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
fig.suptitle('Finance Application: Cleaning Financial News', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: The problem
ax1 = axes[0, 0]
ax1.axis('off')

problem = '''
FINANCIAL NEWS CLEANING CHALLENGE

RAW NEWS DATA CONTAINS:
-----------------------
- HTML tags and formatting
- JavaScript code snippets
- Navigation menus
- Advertisements
- Social media widgets
- Copyright notices
- "Read more" links
- Author bios
- Related articles


EXAMPLE RAW TEXT:
-----------------
"<div class='article'>
<script>analytics.track(...);</script>
<h1>Apple Reports Record Q3 Earnings</h1>
<p class='byline'>By John Smith | Reuters</p>
<p>Apple Inc. (AAPL) reported quarterly earnings...
<a href='...'>Read full story</a>
</p>
<div class='related'>Related: Tech stocks...</div>
<div class='ads'>Advertisement</div>
</div>"


GOAL:
-----
Extract only the meaningful article content:
"Apple Reports Record Q3 Earnings. Apple Inc.
reported quarterly earnings..."


CHALLENGES:
-----------
1. Different formats per source
2. Boilerplate text mixed with content
3. Preserving important financial data
4. Handling encoding issues
'''

ax1.text(0.02, 0.98, problem, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('The Challenge', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Processing statistics
ax2 = axes[0, 1]

# Before/after statistics
categories = ['Characters', 'Words', 'Sentences', 'Unique Terms']
before = [5000, 800, 50, 450]
after = [2000, 300, 25, 200]

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, before, width, label='Before Cleaning', color=MLBLUE, edgecolor='black')
bars2 = ax2.bar(x + width/2, after, width, label='After Cleaning', color=MLGREEN, edgecolor='black')

ax2.set_ylabel('Count')
ax2.set_title('Text Statistics Before/After', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

# Add reduction percentages
for i, (b, a) in enumerate(zip(before, after)):
    reduction = (b - a) / b * 100
    ax2.text(i, max(b, a) + 100, f'-{reduction:.0f}%', ha='center', fontsize=9, color=MLRED)

# Plot 3: Complete news cleaner class
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
COMPLETE FINANCIAL NEWS CLEANER

import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class FinancialNewsCleaner:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))
        # Keep finance-relevant stopwords
        self.stop_words -= {'up', 'down', 'above', 'below', 'not', 'no'}

        # Boilerplate patterns to remove
        self.boilerplate = [
            r'Read more.*',
            r'Click here.*',
            r'Subscribe.*',
            r'Advertisement',
            r'Related:.*',
            r'Copyright.*',
            r'All rights reserved.*',
            r'\\d+ min read',
        ]

    def clean_html(self, html_text):
        """Remove HTML tags and extract text."""
        soup = BeautifulSoup(html_text, 'html.parser')

        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'aside']):
            script.decompose()

        return soup.get_text(separator=' ')

    def clean_text(self, text):
        """Clean extracted text."""
        # Remove boilerplate
        for pattern in self.boilerplate:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove URLs and emails
        text = re.sub(r'https?://\\S+', '', text)
        text = re.sub(r'\\S+@\\S+', '', text)

        # Fix whitespace
        text = re.sub(r'\\s+', ' ', text).strip()

        return text

    def preprocess(self, text, for_nlp=True):
        """Full preprocessing pipeline."""
        # Clean
        text = self.clean_html(text)
        text = self.clean_text(text)

        if not for_nlp:
            return text

        # Tokenize and filter
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens
                  if t.isalpha()
                  and t not in self.stop_words
                  and len(t) > 2]

        return tokens
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=6.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('News Cleaner Class', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Usage example
ax4 = axes[1, 1]
ax4.axis('off')

usage = '''
USAGE EXAMPLE

# Initialize cleaner
cleaner = FinancialNewsCleaner()


# Example 1: Clean HTML article
html_article = """
<html>
<script>trackPageView();</script>
<h1>Apple Reports Q3 Earnings Beat</h1>
<p class="date">October 27, 2023</p>
<p>Apple Inc. (NASDAQ: AAPL) reported third-quarter
earnings that exceeded Wall Street expectations.
The tech giant posted EPS of $1.26 vs $1.19 expected.</p>
<div class="ads">Advertisement</div>
<p>Revenue came in at $89.5 billion, up 5% year-over-year.
Read more about Apple's results.</p>
<footer>Copyright 2023 Reuters</footer>
</html>
"""

# Get clean text
clean_text = cleaner.preprocess(html_article, for_nlp=False)
print(clean_text)
# "Apple Reports Q3 Earnings Beat October 27, 2023 Apple Inc.
#  (NASDAQ: AAPL) reported third-quarter earnings that exceeded
#  Wall Street expectations. The tech giant posted EPS of $1.26
#  vs $1.19 expected. Revenue came in at $89.5 billion, up 5%
#  year-over-year."

# Get tokens for NLP
tokens = cleaner.preprocess(html_article, for_nlp=True)
print(tokens)
# ['apple', 'reports', 'earnings', 'beat', 'october', 'apple',
#  'inc', 'nasdaq', 'aapl', 'reported', 'third', 'quarter',
#  'earnings', 'exceeded', 'wall', 'street', 'expectations',
#  'tech', 'giant', 'posted', 'eps', 'expected', 'revenue',
#  'came', 'billion', 'year']


# Example 2: Batch process news feed
news_articles = load_news_feed()  # Your data source

processed_corpus = []
for article in news_articles:
    tokens = cleaner.preprocess(article['content'])
    processed_corpus.append({
        'id': article['id'],
        'date': article['date'],
        'tokens': tokens,
        'ticker': extract_ticker(article)
    })


# Ready for:
# - Bag of Words
# - TF-IDF
# - Sentiment Analysis
# - Topic Modeling
'''

ax4.text(0.02, 0.98, usage, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Usage Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
